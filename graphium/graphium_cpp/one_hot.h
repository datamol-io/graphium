// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "features.h"

#include <GraphMol/ROMol.h>

#include <stdint.h>

size_t get_one_hot_atom_feature_size(AtomOneHotFeature feature);

template<typename T>
size_t get_one_hot_atom_feature(const GraphData& graph, T* data, AtomOneHotFeature feature, size_t stride);

extern template size_t get_one_hot_atom_feature<int16_t>(const GraphData& graph, int16_t* data, AtomOneHotFeature feature, size_t stride);
extern template size_t get_one_hot_atom_feature<float>(const GraphData& graph, float* data, AtomOneHotFeature feature, size_t stride);
extern template size_t get_one_hot_atom_feature<double>(const GraphData& graph, double* data, AtomOneHotFeature feature, size_t stride);

size_t get_one_hot_bond_feature_size(BondFeature feature);

template<typename T>
size_t get_one_hot_bond_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride);

extern template size_t get_one_hot_bond_feature<int16_t>(const GraphData& graph, int16_t* data, BondFeature feature, size_t stride);
extern template size_t get_one_hot_bond_feature<float>(const GraphData& graph, float* data, BondFeature feature, size_t stride);
extern template size_t get_one_hot_bond_feature<double>(const GraphData& graph, double* data, BondFeature feature, size_t stride);

