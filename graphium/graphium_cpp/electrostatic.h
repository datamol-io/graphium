// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "spectral.h"

#include <stdint.h>
#include <vector>

template<typename T>
void compute_laplacian_pseudoinverse(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& matrix,
    const T* weights = nullptr);

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

template<typename T>
void compute_electrostatic_interactions(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    LaplacianData<T>& data,
    std::vector<T>& laplacian_pseudoinverse,
    std::vector<T>& matrix,
    const T* weights = nullptr);

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
