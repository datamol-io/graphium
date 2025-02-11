// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This header file declares the `compute_electrostatic_interactions`
//!       and `compute_laplacian_pseudoinverse` functions,
//!       defined in electrostatic.cpp and called from features.cpp and commute.cpp

#pragma once

#include "spectral.h"

#include <stdint.h>
#include <vector>

//! Computes the pseudoinverse of the graph Laplacian matrix.
//! Template type `T` can be `float` or `double`.  Implementation is in electrostatic.cpp
//!
//! @param n Number of nodes
//! @param row_starts Array of `n+1` indices into `neighbors`, indicating where each node's
//!                   neighbors start, plus one at the end to indicate the full length of
//!                   `neighbors`
//! @param neighbors Concatenated array of all neighbors of all nodes, in order
//! @param data Cache for the eigendecomposition of the graph Laplacian matrix
//! @param matrix The output pseudoinverse of the graph Laplacian matrix
//! @param weights Optional array of edge weights, in the order corresponding with neighbors.
//!                If null, the edge weights are all 1.
template<typename T>
void compute_laplacian_pseudoinverse(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& matrix,
    const T* weights = nullptr);

// Instantiation declarations of `compute_laplacian_pseudoinverse` for `float` and `double`
// The explicit instantiations are in electrostatic.cpp
extern template void compute_laplacian_pseudoinverse<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<float>& data,
    std::vector<float>& matrix,
    const float* weights);
extern template void compute_laplacian_pseudoinverse<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<double>& data,
    std::vector<double>& matrix,
    const double* weights);

//! Computes the "electrostatic interactions", `P_ij - P_jj`, for each node pair `ij`,
//! where P is the Laplacian pseudoinverse.
//! Template type `T` can be `float` or `double`.  Implementation is in electrostatic.cpp
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
//! @param matrix The output electrostatic interactions for all `n^2` node pairs, i.e. the
//!               pseudoinverse of the graph Laplacian matrix, with the diagonal subtracted from
//!               each column, stored in row-major order.
//! @param weights Optional array of edge weights, in the order corresponding with neighbors.
//!                If null, the edge weights are all 1.
template<typename T>
void compute_electrostatic_interactions(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& laplacian_pseudoinverse,
    std::vector<T>& matrix,
    const T* weights = nullptr);

// Instantiation declarations of `compute_electrostatic_interactions` for `float` and `double`
// The explicit instantiations are in electrostatic.cpp
extern template void compute_electrostatic_interactions<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<float>& data,
    std::vector<float>& laplacian_pseudoinverse,
    std::vector<float>& matrix,
    const float* weights);
extern template void compute_electrostatic_interactions<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<double>& data,
    std::vector<double>& laplacian_pseudoinverse,
    std::vector<double>& matrix,
    const double* weights);
