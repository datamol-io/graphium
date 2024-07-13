// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spectral.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <numeric>

#include "features.h"
#include <ATen/ops/linalg_eig.h>

size_t find_components(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    std::vector<int32_t>& components) {

    int32_t num_components = (n <= 1) ? 1 : 0;
    std::vector<uint32_t> queue;
    if (n > 1) {
        // First, find which nodes are in which component.
        components.resize(n, -1);
        queue.reserve(n);
        for (uint32_t starti = 0; starti < n; ++starti) {
            if (components[starti] >= 0) {
                continue;
            }
            const int32_t component = num_components;
            ++num_components;
            queue.push_back(starti);
            components[starti] = component;
            while (queue.size() != 0) {
                uint32_t current = queue[queue.size()-1];
                queue.resize(queue.size()-1);
                const uint32_t* neighbor_begin = neighbors + row_starts[current];
                const uint32_t* neighbor_end = neighbors + row_starts[current+1];
                for ( ; neighbor_begin != neighbor_end; ++neighbor_begin) {
                    uint32_t neighbor = *neighbor_begin;
                    if (neighbor > starti && components[neighbor] < 0) {
                        components[neighbor] = component;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }
    return size_t(num_components);
}

template<typename T>
void compute_laplacian_eigendecomp_single(const uint32_t n, LaplacianData<T>& data, Normalization normalization) {
    T* matrix = data.matrix_temp.data();
    std::unique_ptr<T[]> matrix_alloc(new T[n * n]);
    std::copy(matrix, matrix + n * n, matrix_alloc.get());

    int64_t dims[2] = { n, n };
    at::Tensor torch_matrix = torch_tensor_from_array<T>(std::move(matrix_alloc), dims, 2, c10::ScalarType::Double);

    // Using linalg_eigh should ensure we get all real eigenvalues and eigenvectors.
    // Arbitrarily choose lower-triangular portion (L)
    auto tuple = at::linalg_eigh(torch_matrix, c10::string_view("L",1));
    at::Tensor eigenvalue_tensor = std::move(std::get<0>(tuple));
    at::Tensor eigenvector_tensor = std::move(std::get<1>(tuple));
    assert(eigenvalue_tensor.ndimension() == 1);
    assert(eigenvector_tensor.ndimension() == 2);
    assert(eigenvalue_tensor.size(0) == n);
    assert(eigenvector_tensor.size(0) == n);
    assert(eigenvector_tensor.size(1) == n);

    // Copy eigenvectors first, because normalization values are in eigenvalues_temp
    data.vectors.clear();
    data.vectors.resize(size_t(n) * n, 0);
    T* vectors = data.vectors.data();
    if (eigenvector_tensor.scalar_type() == c10::ScalarType::Double) {
        const double* const eigenvector_data = eigenvector_tensor.data_ptr<double>();
        for (size_t i = 0; i < size_t(n) * n; ++i) {
            vectors[i] = T(eigenvector_data[i]);
        }

        if (normalization == Normalization::INVERSE) {
            // Convert symmetric case eigenvectors to asymmetric case eigenvectors

            // Scale each row by the factor in eigenvalues_temp
            for (size_t row = 0, i = 0; row < n; ++row) {
                const T factor = data.eigenvalues_temp[row];
                for (size_t col = 0; col < n; ++col, ++i) {
                    vectors[i] *= factor;
                }
                
                // Clear to zero for the summing below
                data.eigenvalues_temp[row] = 0;
            }
            
            // Find each column length
            for (size_t row = 0, i = 0; row < n; ++row) {
                for (size_t col = 0; col < n; ++col, ++i) {
                    const T v = vectors[i];
                    data.eigenvalues_temp[col] += v*v;
                }
            }
            for (size_t col = 0; col < n; ++col) {
                data.eigenvalues_temp[col] = T(1)/std::sqrt(data.eigenvalues_temp[col]);
            }
            
            // Normalize each column
            for (size_t row = 0, i = 0; row < n; ++row) {
                for (size_t col = 0; col < n; ++col, ++i) {
                    vectors[i] *= data.eigenvalues_temp[col];
                }
            }
        }
    }
    else {
        assert(0);
    }

    // Copy eigenvalues
    data.eigenvalues_temp.resize(n);
    if (eigenvalue_tensor.scalar_type() == c10::ScalarType::Double) {
        const double* const eigenvalue_data = eigenvalue_tensor.data_ptr<double>();
        for (size_t i = 0; i < n; ++i) {
            // No adjustment needed to eigenvalues between symmetric and asymmetric
            data.eigenvalues_temp[i] = T(eigenvalue_data[i]);
        }
    }
    else {
        assert(0);
    }

    // Find the sorted order of the eigenvalues
    data.order_temp.resize(n);
    std::iota(data.order_temp.begin(), data.order_temp.end(), 0);
    std::stable_sort(data.order_temp.begin(), data.order_temp.end(),
        [&data](uint32_t i, uint32_t j) -> bool {
            return data.eigenvalues_temp[i] < data.eigenvalues_temp[j];
        }
    );

    // Copy the eigenvalues into the sorted order
    data.eigenvalues.resize(n);
    for (size_t i = 0; i < n; ++i) {
        data.eigenvalues[i] = data.eigenvalues_temp[data.order_temp[i]];
    }

    // Copy the eigenvectors into the sorted order
    std::swap(data.matrix_temp, data.vectors);
    for (size_t row = 0, i = 0; row < n; ++row) {
        const size_t source_row = data.order_temp[row];
        const size_t source_row_start = source_row * n;
        for (size_t col = 0; col < n; ++col, ++i) {
            data.vectors[i] = data.matrix_temp[source_row_start + col];
        }
    }
}

template<typename T>
void compute_laplacian_eigendecomp(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    Normalization normalization,
    LaplacianData<T>& data,
    size_t num_components,
    const std::vector<int32_t>* components,
    const T* weights) {

    // Compute the weight row sums, if applicable, for the diagonal of the laplacian
    if (weights != nullptr) {
        data.eigenvalues_temp.clear();
        data.eigenvalues_temp.resize(n, 0);
        for (uint32_t i = 0; i < n; ++i) {
            const T* weights_begin = weights + row_starts[i];
            const T* weights_end = weights + row_starts[i + 1];
            T sum = T(0);
            for (; weights_begin != weights_end; ++weights_begin) {
                sum += *weights_begin;
            }
            data.eigenvalues_temp[i] = sum;
        }
    }
    data.normalization = normalization;

    // Prepare the laplacian matrix of the graph
    data.matrix_temp.clear();
    data.matrix_temp.resize(size_t(n) * n, 0);
    T* matrix = data.matrix_temp.data();
    if (normalization == Normalization::NONE) {
        for (uint32_t i = 0, outi = 0; i < n; ++i, outi += n) {
            const uint32_t* neighbor_begin = neighbors + row_starts[i];
            const uint32_t* neighbor_end = neighbors + row_starts[i + 1];
            if (weights == nullptr) {
                const uint32_t degree = row_starts[i + 1] - row_starts[i];
                matrix[outi + i] = T(degree);
                for (; neighbor_begin < neighbor_end; ++neighbor_begin) {
                    uint32_t neighbor = *neighbor_begin;
                    matrix[outi + neighbor] = T(-1);
                }
            }
            else {
                matrix[outi + i] = data.eigenvalues_temp[i];
                const T* weights_begin = weights + row_starts[i];
                for (; neighbor_begin < neighbor_end; ++neighbor_begin, ++weights_begin) {
                    uint32_t neighbor = *neighbor_begin;
                    matrix[outi + neighbor] = -(*weights_begin);
                }
            }
        }
    }
    else {
        // The diagonalization of the asymmetric normalization can be computed from the
        // diagonalization of the symmetric normalization, which is faster, so always use symmetric.

        // Find the normalization factor for each node (row or col)
        // These values in eigenvalues_temp are also used inside compute_laplacian_eigendecomp_single
        for (uint32_t node = 0; node < n; ++node) {
            const uint32_t row_degree = row_starts[node + 1] - row_starts[node];
            const T denominator = (weights == nullptr) ? T(row_degree) : data.eigenvalues_temp[node];
            data.eigenvalues_temp[node] = T(1) / std::sqrt(denominator);
        }

        for (uint32_t i = 0, outi = 0; i < n; ++i, outi += n) {
            const uint32_t* neighbor_begin = neighbors + row_starts[i];
            const uint32_t* neighbor_end = neighbors + row_starts[i + 1];
            if (neighbor_begin == neighbor_end) {
                continue;
            }

            // Diagonal is always exactly 1 when normalized (after skipping zero-degree nodes)
            matrix[outi + i] = T(1);

            const T row_factor = data.eigenvalues_temp[i];
            for (; neighbor_begin < neighbor_end; ++neighbor_begin) {
                uint32_t neighbor = *neighbor_begin;
                const T col_factor = data.eigenvalues_temp[neighbor];
                matrix[outi + neighbor] = -row_factor * col_factor;
            }
        }
    }

    if (num_components == 1) {
        compute_laplacian_eigendecomp_single(n, data, normalization);
        return;
    }

    // There are multiple components.
    // To match the original code, handle them separately and
    // pack them into the output.
    
    // data.eigenvalues is length n for the single component case,
    // but to be able to handle this, it needs to be larger, so go with n by n
    data.eigenvalues.clear();
    data.eigenvalues.resize(size_t(n) * n, 0);
    data.vectors.clear();
    data.vectors.resize(size_t(n) * n, 0);
    
    LaplacianData<T> sub_data;
    std::vector<uint32_t> queue;
    for (int32_t component = 0; component < num_components; ++component) {
        // Reuse queue for the indices
        queue.resize(0);
        for (uint32_t i = 0; i < n; ++i) {
            if ((*components)[i] == component) {
                queue.push_back(i);
            }
        }
        
        // Extract the sub-matrix
        const uint32_t sub_n = queue.size();
        sub_data.matrix_temp.resize(size_t(sub_n) * sub_n);
        T* sub_matrix = sub_data.matrix_temp.data();
        for (uint32_t row_index = 0; row_index < sub_n; ++row_index) {
            const uint32_t row = queue[row_index];
            const T*const source_row = matrix + row*size_t(n);
            for (uint32_t col_index = 0; col_index < sub_n; ++col_index) {
                const uint32_t col = queue[col_index];
                *sub_matrix = source_row[col];
                ++sub_matrix;
            }
        }
        
        // Find its eigenvalues and eigenvectors
        compute_laplacian_eigendecomp_single(sub_n, sub_data, normalization);
        
        // Copy the eigenvalues to the output.  The excess is already zeroed out.
        // Unlike the eigenvectors, below, might as well switch to using columns
        // for the eigenvalues, because the caller can handle this case more
        // easily with the single component case this way.
        for (uint32_t row_index = 0; row_index < sub_n; ++row_index) {
            const uint32_t row = queue[row_index];
            T*const dest_row = data.eigenvalues.data() + row*size_t(n);
            for (uint32_t col_index = 0; col_index < sub_n; ++col_index) {
                // Destination data within the row is left justified,
                // NOT distributed based on the component.
                dest_row[col_index] = sub_data.eigenvalues[col_index];
            }
        }

        // Copy the (row) eigenvectors to the output.  The excess is already zeroed out.
        // The caller changes them to column eigenvectors.
        for (uint32_t row_index = 0; row_index < sub_n; ++row_index) {
            // Destination data is top-aligned, NOT distributed
            // based on the component.
            T*const dest_row = data.vectors.data() + row_index*size_t(n);
            const T*const source_row = sub_data.vectors.data() + row_index*size_t(sub_n);
            for (uint32_t col_index = 0; col_index < sub_n; ++col_index) {
                // Columns ARE distributed based on the component.
                const uint32_t col = queue[col_index];
                dest_row[col] = source_row[col_index];
            }
        }
    }
}

template void compute_laplacian_eigendecomp<float>(const uint32_t n, const uint32_t* row_starts, const uint32_t* neighbors, Normalization normalization, LaplacianData<float>& data, size_t num_components, const std::vector<int32_t>* components, const float* weights);
template void compute_laplacian_eigendecomp<double>(const uint32_t n, const uint32_t* row_starts, const uint32_t* neighbors, Normalization normalization, LaplacianData<double>& data, size_t num_components, const std::vector<int32_t>* components, const double* weights);
