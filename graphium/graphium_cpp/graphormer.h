// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <utility>
#include <vector>

template<typename T>
void compute_graphormer_distances(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<T>& all_pairs_distances);

extern template
void compute_graphormer_distances<float>(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<float>& all_pairs_distances);
extern template
void compute_graphormer_distances<double>(
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    std::vector<std::pair<uint32_t, uint32_t>>& queue,
    std::vector<double>& all_pairs_distances);
