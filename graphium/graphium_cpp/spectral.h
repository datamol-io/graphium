// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "features.h"

#include <stdint.h>
#include <vector>

//! Structure for caching eigendecomposition of the graph Laplacian matrix
template<typename T>
struct LaplacianData {
    //! Normalization of the previous eigendecomposition, if computed
    Normalization normalization;

    //! Output/cached eigenvectors of the decomposition, if computed
    std::vector<T> vectors;
    //! Output/cached eigenvalues of the decomposition, if computed
    std::vector<T> eigenvalues;

    //! Temporary arrays used during decomposition
    std::vector<T> matrix_temp;
    std::vector<T> eigenvalues_temp;
    std::vector<uint32_t> order_temp;
};

//! Finds all connected components of the graph, assigning nodes to components.
//!
//! @param n Number of nodes
//! @param row_starts Array of `n+1` indices into `neighbors`, indicating where each node's
//!                   neighbors start, plus one at the end to indicate the full length of
//!                   `neighbors`
//! @param neighbors Concatenated array of all neighbors of all nodes, in order
//! @param components Output array to assign each node an integer indicating which
//!                   component it's in, in the range `[0, num_components)`.  Unused if
//!                   `n < 2`.
//! @return The number of separate connected components found
size_t find_components(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    std::vector<int32_t>& components);

//! Computes the eigendecomposition of the graph Laplacian matrix.
//! This outputs the eigenvalues in `data.eigenvalues` and the eigenvectors in `data.vectors`.
//! The Laplacian matrix is the positive semi-definite matrix `L = D - adj`, where `D` is the
//! diagonal matrix of node degrees, and `adj` is the adjacency matrix.
//! If `normalization` is not `Normalization::NONE`, `L_s = (D^-0.5) L (D^-0.5)` is
//! diagonalized, instead of `L`, and if it's `Normalization::INVERSE`, this is used to compute
//! the decomposition of `L_i = (D^-1) L = (D^-0.5) L_s (D^0.5)`.
//! Template type `T` can be `float` or `double`.  Implementation is in spectral.cpp
//!
//! @param n Number of nodes
//! @param row_starts Array of `n+1` indices into `neighbors`, indicating where each node's
//!                   neighbors start, plus one at the end to indicate the full length of
//!                   `neighbors`
//! @param neighbors Concatenated array of all neighbors of all nodes, in order
//! @param normalization Whether and how to normalize the Laplacian matrix before diagonalizing.
//!                      This is recorded into `data.normalization` for caching use.
//! @param data Output and temporary arrays for the eigendecomposition of the graph Laplacian
//!             matrix
//! @param num_components The number of connected components, if separately diagonalizing
//!                       each component, else 1 to diagonalize the entire graph as a whole
//! @param components Optional array of length `n`, where each node's integer indicates which
//!                   component it is in, in the range `[0, num_components)`
//! @param weights Optional array of edge weights, in the order corresponding with neighbors.
//!                If null, the edge weights are all 1.
template<typename T>
void compute_laplacian_eigendecomp(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    Normalization normalization,
    LaplacianData<T>& data,
    size_t num_components,
    const int32_t* components,
    const T* weights = nullptr);

// Instantiation declarations of `compute_laplacian_eigendecomp` for `float` and `double`
// The explicit instantiations are in spectral.cpp
extern template void compute_laplacian_eigendecomp<float>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    Normalization normalization,
    LaplacianData<float>& data,
    size_t num_components,
    const int32_t* components,
    const float* weights);
extern template void compute_laplacian_eigendecomp<double>(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    Normalization normalization,
    LaplacianData<double>& data,
    size_t num_components,
    const int32_t* components,
    const double* weights);
