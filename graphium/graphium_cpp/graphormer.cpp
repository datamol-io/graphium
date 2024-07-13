// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "graphormer.h"

#include <algorithm>
#include <stdint.h>
#include <utility>
#include <vector>

template<typename T>
void compute_graphormer_distances(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t,uint32_t>>& queue,
    std::vector<T>& all_pairs_distances) {

    // Compute all pairs shortest paths.
    // Because this is a sparse graph treated as having unweighted edges,
    // BFS on each node is faster than Dijkstra's or Floyd-Warshall's.

    if (queue.capacity() == 0) {
        queue.reserve(n);
    }

    all_pairs_distances.resize(size_t(n) * n);
    std::fill(all_pairs_distances.begin(), all_pairs_distances.end(), T(-1));

    for (uint32_t start_index = 0; start_index < n; ++start_index) {
        queue.resize(0);
        size_t queue_head = 0;
        queue.push_back({ start_index,0 });
        T* const distances = all_pairs_distances.data() + start_index * n;
        while (queue.size() != queue_head) {
            auto [current_node, current_distance] = queue[queue_head];
            ++queue_head;

            if (distances[current_node] != T(-1)) {
                continue;
            }

            distances[current_node] = T(current_distance);

            ++current_distance;

            const uint32_t* neighbor_start = neighbors + neighbor_starts[current_node];
            const uint32_t* neighbor_end = neighbors + neighbor_starts[current_node+1];
            for (; neighbor_start != neighbor_end; ++neighbor_start) {
                queue.push_back({ *neighbor_start,current_distance });
            }
        }
    }
}

// Explicit instantiations for float and double
template
void compute_graphormer_distances<float>(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<float>& all_pairs_distances);
template
void compute_graphormer_distances<double>(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<double>& all_pairs_distances);
