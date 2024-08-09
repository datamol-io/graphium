// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "features.h"

#include <GraphMol/ROMol.h>

#include <stdint.h>

template<typename T>
void get_atom_float_feature(const GraphData& graph, T* data, AtomFloatFeature feature, size_t stride, bool offset_carbon = true);

extern template void get_atom_float_feature<int16_t>(const GraphData& graph, int16_t* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
extern template void get_atom_float_feature<float>(const GraphData& graph, float* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
extern template void get_atom_float_feature<double>(const GraphData& graph, double* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);

template<typename T>
void get_bond_float_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride);

extern template void get_bond_float_feature<int16_t>(const GraphData& graph, int16_t* data, BondFeature feature, size_t stride);
extern template void get_bond_float_feature<float>(const GraphData& graph, float* data, BondFeature feature, size_t stride);
extern template void get_bond_float_feature<double>(const GraphData& graph, double* data, BondFeature feature, size_t stride);

// This table is from the Group column of graphium/features/periodic_table.csv
constexpr uint8_t atomicNumToGroupTable[] = {
         1, 18,  1,  2, 13, 14, 15, 16, 17,
    18,  1,  2, 13, 14, 15, 16, 17, 18,  1,
     2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17, 18,  1,  2,  3,
     4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
    14, 15, 16, 17, 18,  1,  2,  3, 19, 19,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19,  4,  5,  6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17, 18,  1,  2,  3,
    19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    19, 19, 19, 19,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18,
};
constexpr size_t groupCount = 19;

// This table is from the Period column of graphium/features/periodic_table.csv
constexpr uint8_t atomicNumToPeriodTable[] = {
       1, 1, 2, 2, 2, 2, 2, 2, 2,
    2, 3, 3, 3, 3, 3, 3, 3, 3, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7
};
constexpr size_t periodCount = 7;
