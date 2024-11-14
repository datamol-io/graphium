// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
@file
*/


#include "random_walk.h"

#include <assert.h>
#include <cmath>
#include <stdint.h>
#include <utility>
#include <vector>

//! Multiplies the dense `n` by `n` matrix `in_matrix` by the sparse `n` by `n` matrix in CSC
//! format (transpose of CSR format) represented by `neighbor_starts`, `neighbors`, and
//! `col_major_weights`, writing the results into `out_matrix`.
template<typename T>
void multiply_dense_by_sparse(uint32_t n, T* out_matrix, const T* in_matrix, const uint32_t* neighbor_starts, const uint32_t* neighbors, const T* col_major_weights) {
    for (uint32_t row = 0; row < n; ++row) {
        T* out_row_start = out_matrix + row * n;
        const T* in_row_start = in_matrix + row * n;
        for (uint32_t col = 0; col < n; ++col) {
            T sum = T(0);
            // The adjacency is symmetric, so rows and cols are swappable there,
            // but the weights might not be, so for fast access, we want column major weights.
            const uint32_t* neighbors_start = neighbors + neighbor_starts[col];
            const uint32_t* neighbors_end = neighbors + neighbor_starts[col+1];
            const T* weights_start = col_major_weights + neighbor_starts[col];
            for (; neighbors_start != neighbors_end; ++neighbors_start, ++weights_start) {
                sum += *weights_start * in_row_start[*neighbors_start];
            }
            out_row_start[col] = sum;
        }
    }
}

// Computes random walk data about the graph, either probabilities or transfer amounts
// after certain numbers of steps, outputting the values to `output`.
// See the declaration in random_walk.h for more details.
template<typename T>
void compute_rwse(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<T>& output,
    int space_dim) {

    // Cast one n to size_t to avoid integer overflow if n >= 65536
    if (option == RandomWalkDataOption::PROBABILITIES) {
        output.resize(num_powers * size_t(n));
    }
    else {
        output.resize(num_powers * size_t(n) * n);
    }

    if (num_powers == 0) {
        return;
    }
    if (n == 1) {
        // Special case: All ones for single node, matching original code
        for (uint32_t i = 0; i < output.size(); ++i) {
            output[i] = T(1);
        }
        return;
    }

    // Initialize this to represent column major D^-1 * adj
    std::vector<T> col_major_weights;
    col_major_weights.resize(neighbor_starts[n]);
    for (uint32_t col = 0, i = 0; col < n; ++col) {
        const uint32_t* neighbor_start = neighbors + neighbor_starts[col];
        const uint32_t* neighbor_end = neighbors + neighbor_starts[col+1];
        for (; neighbor_start != neighbor_end; ++neighbor_start, ++i) {
            const uint32_t neighbor = *neighbor_start;
            uint32_t neighbor_degree = neighbor_starts[neighbor + 1] - neighbor_starts[neighbor];
            T degree_inv = (neighbor_degree == 0) ? T(0) : T(1) / T(neighbor_degree);
            col_major_weights[i] = degree_inv;
        }
    }

    // Space for 2 matrices, to alternate between them
    std::vector<T> matrix;
    matrix.resize(2 * size_t(n) * n, T(0));
    T* matrix0 = matrix.data();
    T* matrix1 = matrix.data() + size_t(n) * n;
    uint64_t current_power = 0;
    // Initialize current matrix to identity matrix
    for (size_t i = 0, diag_index = 0; i < n; ++i, diag_index += (n+1)) {
        matrix0[diag_index] = T(1);
    }

    for (uint32_t power_index = 0; power_index < num_powers; ++power_index) {
        const uint64_t target_power = powers[power_index];
        assert(target_power >= current_power);
        while (target_power > current_power) {
            std::swap(matrix0, matrix1);
            multiply_dense_by_sparse(n, matrix0, matrix1, neighbor_starts, neighbors, col_major_weights.data());
            ++current_power;
        }

        // Copy results to output
        if (option == RandomWalkDataOption::PROBABILITIES) {
            const T scale_factor = (space_dim == 0) ? T(1) : T(std::pow(T(target_power), T(0.5) * T(space_dim)));
            // Just copy the diagonal values
            for (size_t i = 0, diag_index = 0; i < n; ++i, diag_index += (n + 1)) {
                output[i * num_powers + power_index] = scale_factor * matrix0[diag_index];
            }
        }
        else {
            // Copy transition probabilities, making sure the dimensions are correct, because matrix0 isn't symmetric.
            // Least significant dimension is num_powers
            // Middle dimension is the columns across a single row of matrix0
            // Most significant dimension is the rows of the matrix0
            const size_t row_stride = num_powers * size_t(n);
            for (size_t row = 0, i = 0; row < n; ++row) {
                for (size_t col = 0; col < n; ++col, ++i) {
                    output[row * row_stride + col * num_powers + power_index] = matrix0[i];
                }
            }
        }
    }
}

// Explicit instantiations of `compute_rwse` for `float` and `double`
template void compute_rwse<float>(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<float>& output,
    int space_dim);
template void compute_rwse<double>(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<double>& output,
    int space_dim);
