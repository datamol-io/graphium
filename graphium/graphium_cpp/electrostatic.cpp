// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "electrostatic.h"

#include "spectral.h"

#include <limits>
#include <stdint.h>
#include <vector>

// Computes the pseudoinverse of the graph Laplacian, outputting to `matrix`.
// See the declaration in electrostatic.h for more details.
template<typename T>
void compute_laplacian_pseudoinverse(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& matrix,
    const T* weights) {

    // If we've already computed the eigendecomposition with the correct normalization,
    // skip recomputing it.
    if (data.eigenvalues.size() != n || data.normalization != Normalization::NONE) {
        compute_laplacian_eigendecomp(n, row_starts, neighbors, Normalization::NONE, data, 1, nullptr, weights);
    }

    // Allocate the space for the output and initialize to zero.
    // The clear() call is so that resize() initializes all values to zero.
    matrix.clear();
    matrix.resize(size_t(n) * n, T(0));
    const T maxEigenvalue = data.eigenvalues.back();
    // zero_threshold is an estimate of how accurately the diagonalization
    // algorithm determines eigenvalues close to zero.  Anything smaller
    // should be considered zero for the pseudoinverse.
    const T eigendecomp_relative_threshold = T(1e-6);
    const T zero_threshold = n * eigendecomp_relative_threshold * maxEigenvalue;
    for (size_t eigenIndex = 0; eigenIndex < n; ++eigenIndex) {
        // This is a positive semi-definite matrix, so we don't need to take the absolute value
        // when checking the threshold.
        if (data.eigenvalues[eigenIndex] < zero_threshold) {
            continue;
        }
        const T eigenvalueInverse = T(1) / data.eigenvalues[eigenIndex];
        const T* const eigenvector = data.vectors.data() + eigenIndex * n;
        for (size_t row = 0, i = 0; row < n; ++row) {
            for (size_t col = 0; col < n; ++col, ++i) {
                const T value = eigenvalueInverse * eigenvector[row] * eigenvector[col];
                matrix[i] += value;
            }
        }
    }
}

// Explicit instantiations of `compute_laplacian_pseudoinverse` for `float` and `double`
template void compute_laplacian_pseudoinverse<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<float>& data,
    std::vector<float>& matrix,
    const float* weights);
template void compute_laplacian_pseudoinverse<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<double>& data,
    std::vector<double>& matrix,
    const double* weights);

// Computes the "electrostatic interactions" between each pair of nodes, outputting to `matrix`.
// See the declaration in electrostatic.h for more details.
template<typename T>
void compute_electrostatic_interactions(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& laplacian_pseudoinverse,
    std::vector<T>& matrix,
    const T* weights) {

    if (laplacian_pseudoinverse.size() == 0) {
        compute_laplacian_pseudoinverse(n, row_starts, neighbors, data, laplacian_pseudoinverse, weights);
    }

    // Allocate the memory for the output
    matrix.resize(n * n);

    // Subtract the diagonal value from each column
    for (size_t row = 0, i = 0; row < n; ++row) {
        for (size_t col = 0, diag_index = 0; col < n; ++col, ++i, diag_index += (n+1)) {
            matrix[i] = laplacian_pseudoinverse[i] - laplacian_pseudoinverse[diag_index];
        }
    }
}

// Explicit instantiations of `compute_electrostatic_interactions` for `float` and `double`
template void compute_electrostatic_interactions<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<float>& data,
    std::vector<float>& laplacian_pseudoinverse,
    std::vector<float>& matrix,
    const float* weights);
template void compute_electrostatic_interactions<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<double>& data,
    std::vector<double>& laplacian_pseudoinverse,
    std::vector<double>& matrix,
    const double* weights);
