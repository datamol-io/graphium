// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <vector>

enum class RandomWalkDataOption {
    PROBABILITIES,
    MATRIX
};

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

extern template
void compute_rwse<float>(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<float>& output,
    int space_dim);
extern template
void compute_rwse<double>(
    const uint32_t num_powers,
    const uint64_t* powers,
    const uint32_t n,
    const uint32_t* neighbor_starts,
    const uint32_t* neighbors,
    RandomWalkDataOption option,
    std::vector<double>& output,
    int space_dim);
