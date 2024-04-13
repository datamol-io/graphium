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

// This outputs the eigenvalues in data.eigenvalues and the eigenvectors in data.vectors
template<typename T>
void compute_laplacian_eigendecomp(
    const uint32_t n,
    const uint32_t* row_starts,
    const uint32_t* neighbors,
    Normalization normalization,
    LaplacianData<T>& data,
    bool disconnected_comp = false,
    const T* weights = nullptr);

extern template void compute_laplacian_eigendecomp<float>(const uint32_t n, const uint32_t* row_starts, const uint32_t* neighbors, Normalization normalization, LaplacianData<float>& data, bool disconnected_comp, const float* weights);
extern template void compute_laplacian_eigendecomp<double>(const uint32_t n, const uint32_t* row_starts, const uint32_t* neighbors, Normalization normalization, LaplacianData<double>& data, bool disconnected_comp, const double* weights);
