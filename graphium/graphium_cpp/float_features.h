// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "features.h"

#include <GraphMol/ROMol.h>

#include <stdint.h>

//! Fills in a particular atom float `feature` into `data`, for all atoms.
//! Template type `T` can be `int16_t` (FP16), `float`, or `double`.
//! Implementation is in float_features.cpp
//!
//! @param graph Molecule containing the source data
//! @param data Destination array, pointing to the first atom's value for this
//!             feature to be filled in.  Each atom's data for this feature is just 1 value,
//!	            but because different features are interleaved, the values for
//!             each atom are spaced `stride` values apart.
//! @param feature The atom feature to write into `data`
//! @param stride The number of values from the beginning of one atom's data to the beginning
//!               of the next atom's data, which may include values for other features
//! @param offset_carbon If true (the default), a reference value for carbon is subtracted,
//!                      so that carbon atoms would usually have value zero, if applicable.
//! @see AtomFloatFeature
template<typename T>
void get_atom_float_feature(const GraphData& graph, T* data, AtomFloatFeature feature, size_t stride, bool offset_carbon = true);

// Instantiation declarations of `get_atom_float_feature` for `int16_t` (FP16),
// `float` (FP32), and `double` (FP64). The explicit instantiations are in float_features.cpp
extern template void get_atom_float_feature<int16_t>(const GraphData& graph, int16_t* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
extern template void get_atom_float_feature<float>(const GraphData& graph, float* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
extern template void get_atom_float_feature<double>(const GraphData& graph, double* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);

//! Fills in a particular bond float `feature` into `data`, for all bonds.
//! Template type `T` can be `int16_t` (FP16), `float`, or `double`.
//! Implementation is in float_features.cpp
//!
//! @param graph Molecule containing the source data
//! @param data Destination array, pointing to the first bond's value for this
//!             feature to be filled in.  Each bond's data for this feature is just 1 value,
//!	            but because different features are interleaved, the values for
//!             each bond are spaced `stride` values apart.
//! @param feature The bond feature to write into `data`
//! @param stride The number of values from the beginning of one bond's data to the beginning
//!               of the next bond's data, which may include values for other features
//! @see BondFeature
template<typename T>
void get_bond_float_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride);

// Instantiation declarations of `get_bond_float_feature` for `int16_t` (FP16),
// `float` (FP32), and `double` (FP64). The explicit instantiations are in float_features.cpp
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
