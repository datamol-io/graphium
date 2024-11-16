// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file This header file declares functions for one-hot atom and bond features,
//!       defined in one_hot.cpp and called from features.cpp

#pragma once

#include "features.h"

#include <GraphMol/ROMol.h>

#include <stdint.h>


//! Returns the number of values per atom, required by `feature` in `get_one_hot_atom_feature`'s
//! `data` argument.  Implementation is in one_hot.cpp
size_t get_one_hot_atom_feature_size(AtomOneHotFeature feature);

//! Fills in a particular atom `feature`'s one-hot encoding into `data`, for all atoms.
//! Template type `T` can be `int16_t` (FP16), `float`, or `double`.
//! Implementation is in one_hot.cpp
//!
//! @param graph Molecule containing the source data to one-hot encode
//! @param data Destination array, pointing to the first atom's one-hot values for this
//!             feature to be filled in.  Each atom's data for this feature is
//!	            `get_one_hot_atom_feature_size(feature)` values long, but because different
//!             features are interleaved, the beginnings of the data for each atom are spaced
//!             `stride` values apart, which will be greater if there are other features.
//! @param feature The atom feature to one-hot encode (i.e. all zeros except a single one
//!	               whose index represents the feature value) into `data`
//! @param stride The number of values from the beginning of one atom's data to the beginning
//!               of the next atom's data, which may include values for other features
//! @return The number of values per atom, i.e. `get_one_hot_atom_feature_size(feature)`
//! @see AtomOneHotFeature
//! @see get_one_hot_atom_feature_size
template<typename T>
size_t get_one_hot_atom_feature(const GraphData& graph, T* data, AtomOneHotFeature feature, size_t stride);

// Instantiation declarations of `get_one_hot_atom_feature` for `int16_t` (FP16),
// `float` (FP32), and `double` (FP64). The explicit instantiations are in one_hot.cpp
extern template size_t get_one_hot_atom_feature<int16_t>(const GraphData& graph, int16_t* data, AtomOneHotFeature feature, size_t stride);
extern template size_t get_one_hot_atom_feature<float>(const GraphData& graph, float* data, AtomOneHotFeature feature, size_t stride);
extern template size_t get_one_hot_atom_feature<double>(const GraphData& graph, double* data, AtomOneHotFeature feature, size_t stride);

//! Returns the number of values required by `feature` in `get_one_hot_bond_feature`'s
//! `data` argument.  Implementation is in one_hot.cpp
size_t get_one_hot_bond_feature_size(BondFeature feature);

//! Fills in a particular bond `feature`'s one-hot encoding into `data`, for all bonds.
//! Template type `T` can be `int16_t` (FP16), `float`, or `double`.
//! Implementation is in one_hot.cpp
//!
//! @param graph Molecule containing the source data to one-hot encode
//! @param data Destination array, pointing to the first bond's one-hot values for this
//!             feature to be filled in.  Each bond's data for this feature is
//!	            `get_one_hot_bond_feature_size(feature)` values long, but because different
//!             features are interleaved, the beginnings of the data for each bond are spaced
//!             `stride` values apart, which will be greater if there are other features.
//! @param feature The bond feature to one-hot encode (i.e. all zeros except a single one
//!	               whose index represents the feature value) into `data`
//! @param stride The number of values from the beginning of one bond's data to the beginning
//!               of the next bond's data, which may include values for other features
//! @return The number of values per bond, i.e. `get_one_hot_bond_feature_size(feature)`
//! @see BondFeature
//! @see get_one_hot_bond_feature_size
template<typename T>
size_t get_one_hot_bond_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride);

// Instantiation declarations of `get_one_hot_bond_feature` for `int16_t` (FP16),
// `float` (FP32), and `double` (FP64). The explicit instantiations are in one_hot.cpp
extern template size_t get_one_hot_bond_feature<int16_t>(const GraphData& graph, int16_t* data, BondFeature feature, size_t stride);
extern template size_t get_one_hot_bond_feature<float>(const GraphData& graph, float* data, BondFeature feature, size_t stride);
extern template size_t get_one_hot_bond_feature<double>(const GraphData& graph, double* data, BondFeature feature, size_t stride);

