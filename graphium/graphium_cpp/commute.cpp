// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "commute.h"

#include "electrostatic.h"
#include "spectral.h"

#include <stdint.h>
#include <vector>

template<typename T>
void compute_commute_distances(
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

    T full_sum = T(0);
    if (weights != nullptr) {
        for (size_t i = 0, weights_size = row_starts[n]; i < weights_size; ++i) {
            full_sum += weights[i];
        }
    }
    else {
        // Unweighted, so just twice the unique edge count
        // (each edge appears twice in neighbors)
        full_sum = T(row_starts[n]);
    }

    matrix.resize(n * n);

    for (size_t row = 0, row_diag_index = 0, i = 0; row < n; ++row, row_diag_index += (n + 1)) {
        for (size_t col = 0, col_diag_index = 0; col < n; ++col, ++i, col_diag_index += (n + 1)) {
            matrix[i] = full_sum * (
                laplacian_pseudoinverse[row_diag_index]
                + laplacian_pseudoinverse[col_diag_index]
                - 2 * laplacian_pseudoinverse[row*n + col]);
        }
    }
}

template
void compute_commute_distances<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<float>& data,
    std::vector<float>& laplacian_pseudoinverse,
    std::vector<float>& matrix,
    const float* weights);
template
void compute_commute_distances<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<double>& data,
    std::vector<double>& laplacian_pseudoinverse,
    std::vector<double>& matrix,
    const double* weights);
