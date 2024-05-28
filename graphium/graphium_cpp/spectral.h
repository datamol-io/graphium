// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "features.h"

#include <stdint.h>
#include <vector>

template<typename T>
struct LaplacianData {
    Normalization normalization;

    std::vector<T> vectors;
    std::vector<T> eigenvalues;

    std::vector<T> matrix_temp;
    std::vector<T> eigenvalues_temp;
    std::vector<uint32_t> order_temp;
};

size_t find_components(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    std::vector<int32_t>& components);

// This outputs the eigenvalues in data.eigenvalues and the eigenvectors in data.vectors
template<typename T>
void compute_laplacian_eigendecomp(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    Normalization normalization,
    LaplacianData<T>& data,
    size_t num_components,
    const std::vector<int32_t>* components,
    const T* weights = nullptr);

extern template void compute_laplacian_eigendecomp<float>(const uint32_t n, const uint32_t* row_starts, const uint32_t* neighbors, Normalization normalization, LaplacianData<float>& data, size_t num_components, const std::vector<int32_t>* components, const float* weights);
extern template void compute_laplacian_eigendecomp<double>(const uint32_t n, const uint32_t* row_starts, const uint32_t* neighbors, Normalization normalization, LaplacianData<double>& data, size_t num_components, const std::vector<int32_t>* components, const double* weights);
