// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This header file declares the `compute_graphormer_distances` function,
//!       defined in graphormer.cpp and called from features.cpp

#pragma once

#include <stdint.h>
#include <utility>
#include <vector>

//! Computes the shortest path distance, along edges, between all pairs of nodes, outputting to
//! `all_pairs_distances`.
//! Template type `T` can be `float` or `double`.  Implementation is in graphormer.cpp
//!
//! @param n Number of nodes
//! @param neighbor_starts Array of `n+1` indices into `neighbors`, indicating where each node's
//!                        neighbors start, plus one at the end to indicate the full length of
//!                        `neighbors`
//! @param neighbors Concatenated array of all neighbors of all nodes, in order
//! @param queue vector used for temporary storage internally
//! @param all_pairs_distances This will be filled with the unweighted lengths of the shortest
//!                            path between each pair of nodes.
template<typename T>
void compute_graphormer_distances(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<T>& all_pairs_distances);

// Instantiation declarations of `compute_graphormer_distances` for `float` and `double`
// The explicit instantiations are in graphormer.cpp
extern template void compute_graphormer_distances<float>(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<float>& all_pairs_distances);
extern template void compute_graphormer_distances<double>(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<double>& all_pairs_distances);
