// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <vector>

//! Options for the `option` parameter of `compute_rwse` function
enum class RandomWalkDataOption {
    PROBABILITIES,
    MATRIX
};

//! Computes random walk data about the graph, either probabilities or transfer amounts
//! after certain numbers of steps, outputting the values to `output`.
//! Template type `T` can be `float` or `double`.  Implementation is in random_walk.cpp
//! 
//! The adjacency (neighbor_starts and neighbors) must be symmetric.
//!
//! @param num_powers The length of `powers`
//! @param powers Array of `num_powers` integers, with each one indicating how many steps at
//!               which to output the probabilities or transfer amounts.
//!               This *must* be in increasing order.
//! @param n Number of nodes
//! @param neighbor_starts Array of `n+1` indices into `neighbors`, indicating where each node's
//!                        neighbors start, plus one at the end to indicate the full length of
//!                        `neighbors`
//! @param neighbors Concatenated array of all neighbors of all nodes, in order
//! @param option Whether to output `n` probabilities for each power or a matrix of `n^2`
//!               transfer amounts for each power
//! @param output Array of values to be filled with `n * num_powers` probabilities or
//!               `n^2 * num_powers` transfer amounts, as if this is a 2D array
//!               `[n][num_powers]` or 3D array `[rows][cols][num_powers]`, respectively
//! @param space_dim Optional parameter to scale probabilities by `power^(0.5*space_dim)`.
//!                  Default of zero corresponds with not scaling the probabilities.
template<typename T>
void compute_rwse(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<T>& output,
    int space_dim = 0);

// Instantiation declarations of `compute_rwse` for `float` and `double`
// The explicit instantiations are in random_walk.cpp
extern template void compute_rwse<float>(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<float>& output,
    int space_dim);
extern template void compute_rwse<double>(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<double>& output,
    int space_dim);
