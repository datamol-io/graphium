// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This header file declares the `compute_commute_distances` function,
//!       defined in commute.cpp and called from features.cpp

#pragma once

#include "spectral.h"

#include <stdint.h>
#include <vector>

//! Computes the "commute distance", `2*E*(P_ii + P_jj - 2*P_ij)`, for each node pair `ij`,
//! where P is the Laplacian pseudoinverse and E is the total number of unique edges.
//! Template type `T` can be `float` or `double`.  Implementation is in commute.cpp
//!
//! @param n Number of nodes
//! @param row_starts Array of `n+1` indices into `neighbors`, indicating where each node's
//!                   neighbors start, plus one at the end to indicate the full length of
//!                   `neighbors`
//! @param neighbors Concatenated array of all neighbors of all nodes, in order
//! @param data Cache for the eigendecomposition of the graph Laplacian matrix
//! @param laplacian_pseudoinverse If empty, this will be filled with the pseudoinverse of the
//!                                graph Laplacian matrix, else its contents will be assumed to
//!                                contain the cached pseudoinverse of the graph Laplacian
//! @param matrix The output commute distances for all `n^2` node pairs
//! @param weights Optional array of edge weights, in the order corresponding with neighbors.
//!                If non-null, the distances will be scaled by the sum of all weights, instead
//!                of `2*E`.
template<typename T>
void compute_commute_distances(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& laplacian_pseudoinverse,
    std::vector<T>& matrix,
    const T* weights = nullptr);

// Instantiation declarations of `compute_commute_distances` for `float` and `double`
// The explicit instantiations are in commute.cpp
extern template void compute_commute_distances<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<float>& data,
    std::vector<float>& laplacian_pseudoinverse,
    std::vector<float>& matrix,
    const float* weights);
extern template void compute_commute_distances<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<double>& data,
    std::vector<double>& laplacian_pseudoinverse,
    std::vector<double>& matrix,
    const double* weights);
