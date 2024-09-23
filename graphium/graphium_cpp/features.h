// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <stdint.h>
#include <type_traits>

// Torch tensor headers
#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>

// PyBind and Torch headers
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

//! Levels at which features or labels can be associated
//! String names are in `feature_level_to_enum` in features.cpp
enum class FeatureLevel {
    NODE,       //!< Values for each node (atom)
    EDGE,       //!< Values for each edge (bond)
    NODEPAIR,   //!< Values for each pair of nodes (pair of atoms), even if no edge (bond)
    GRAPH       //!< Values for whole molecule
};

//! Features for use by `get_atom_float_feature` in float_features.cpp
//! String names are in `atom_float_name_to_enum` in features.cpp
enum class AtomFloatFeature {
    ATOMIC_NUMBER,
    MASS,
    VALENCE,
    IMPLICIT_VALENCE,
    HYBRIDIZATION,
    CHIRALITY,
    AROMATIC,
    IN_RING,
    MIN_RING,
    MAX_RING,
    NUM_RING,
    DEGREE,
    RADICAL_ELECTRON,
    FORMAL_CHARGE,
    VDW_RADIUS,
    COVALENT_RADIUS,
    ELECTRONEGATIVITY,
    IONIZATION,
    MELTING_POINT,
    METAL,
    GROUP,
    PERIOD,
    SINGLE_BOND,
    AROMATIC_BOND,
    DOUBLE_BOND,
    TRIPLE_BOND,
    IS_CARBON,
    UNKNOWN
};

//! Features for use by `get_one_hot_atom_feature` in one_hot.cpp
//! String names are in `atom_onehot_name_to_enum` in features.cpp
enum class AtomOneHotFeature {
    ATOMIC_NUM,         //!< Selected atomic numbers specified in `atomicNumList` in one_hot.cpp
    DEGREE,             //!< Number of explicit neighboring atoms
    VALENCE,            //!< Total valence of the atom
    IMPLICIT_VALENCE,   //!< Implicit valence of the atom
    HYBRIDIZATION,      //!< Hybridizations specified in `hybridizationList` in one_hot.cpp
    CHIRALITY,          //!< "R", anything other value ("S") or no value, and an extra
                        //!< chirality-related value (independent of the other two, so can
                        //!< have a 2nd one value)
    PHASE,              //!< Specified by `ElementPhase` and `atomicNumToPhase` in one_hot.cpp
    TYPE,               //!< Specified by `ElementType` and `atomicNumToType` in one_hot.cpp
    GROUP,              //!< Specified by `atomicNumToGroupTable` in float_features.h
    PERIOD,             //!< Specified by `atomicNumToPeriodTable` in float_features.h
    UNKNOWN             //!< Sentinel value.  Do not use.
};

//! Features for use by `get_one_hot_bond_feature` in one_hot.cpp (if ends in `ONE_HOT`), and
//! `get_bond_float_feature` in float_features.cpp
//! String names are in `bond_name_to_enum` in features.cpp
enum class BondFeature {
    TYPE_FLOAT,         //!< Bond type as a float, e.g. 2.0 for double, 1.5 for aromatic
    TYPE_ONE_HOT,       //!< Selected bond types specified in `bondTypeList` in one_hot.cpp
    IN_RING,            //!< 1.0 if the bond is in at least one ring, else 0.0
    CONJUGATED,         //!< 1.0 if the bond is conjugated, else 0.0
    STEREO_ONE_HOT,     //!< Selected bond stereo values specified in `bondStereoList` in
                        //!< one_hot.cpp
    CONFORMER_BOND_LENGTH,//!< Length of the bond from a conformer (either first or computed)
    ESTIMATED_BOND_LENGTH,//!< Length of the bond estimated with a fast heuristic
    UNKNOWN             //!< Sentinel value.  Do not use.
};

//! Supported "positional" features
//! String names are in `positional_name_to_enum` in features.cpp
enum class PositionalFeature {
    LAPLACIAN_EIGENVEC, //!< See `compute_laplacian_eigendecomp` in spectral.cpp
    LAPLACIAN_EIGENVAL, //!< See `compute_laplacian_eigendecomp` in spectral.cpp
    RW_RETURN_PROBS,    //!< See `compute_rwse` in random_walk.cpp
    RW_TRANSITION_PROBS,//!< See `compute_rwse` in random_walk.cpp
    ELECTROSTATIC,      //!< See `compute_electrostatic_interactions` in electrostatic.cpp
    COMMUTE,            //!< See `compute_commute_distances` in commute.cpp
    GRAPHORMER          //!< See `compute_graphormer_distances` in graphormer.cpp
};

//! Options for normalization of graph Laplacian matrix in positional features.
//! Not to be confused with the normalization of label data in `prepare_and_save_data`.
//! String names are in `normalization_to_enum` in features.cpp
enum class Normalization {
    NONE,       //!< Leaves the matrix unnormalized: `L = D - adj`
    SYMMETRIC,  //!< Corresponds with `L_s = (D^-0.5) L (D^-0.5)`
    INVERSE     //!< Corresponds with `L_i = (D^-1) L`
};

//! Options for handling NaN or infinite values, passed from Python to `featurize_smiles` in
//! features.cpp.  Masking is done in `mask_nans` in features.h
enum class MaskNaNStyle {
    NONE,   //!< Ignore (keep) NaN values
    REPORT, //!< (default behaviour) Count NaN values and report that with the index of the
            //!< first tensor that contained NaNs
    REPLACE //!< Replace NaN values with a specific value (defaults to zero)
};

//! Class for storing all supported options of all positional features,
//! even ones that are mutually exclusive with each other.
struct PositionalOptions {
    PositionalFeature feature;
    FeatureLevel level;

    //! Powers used by `PositionalFeature::RW_RETURN_PROBS` and `RW_TRANSITION_PROBS`
    std::vector<uint32_t> rw_powers;
    int rw_space_dim = 0;

    uint32_t laplacian_num_pos = 8;
    Normalization laplacian_normalization = Normalization::NONE;
    bool laplacian_disconnected_comp = true;
};

//! Class to help supporting `int16_t` as if it's a 16-bit floating-point (FP16) type,
//! while still supporting `float` (FP32) and `double` (FP64).
template<typename T>
struct FeatureValues {};

//! Explicit instantiation of `FeatureValues` for `int16_t` as if it's a 16-bit
//! floating-point (FP16) type.
template<> struct FeatureValues<int16_t> {
    static constexpr int16_t zero = 0x0000;
    static constexpr int16_t one = 0x3C00;
    static constexpr int16_t nan_value = 0x7C01;

    template<typename T>
    static int16_t convertToFeatureType(T inputType) {
        static_assert(std::is_floating_point_v<T>);
        return c10::detail::fp16_ieee_from_fp32_value(float(inputType));
    }

    static constexpr bool is_finite(int16_t v) {
        // If the exponent bits are the maximum value, v is infinite or NaN
        return (v & 0x7C00) != 0x7C00;
    }

    using MathType = float;
};
//! Explicit instantiation of `FeatureValues` for `float` (FP32)
template<> struct FeatureValues<float> {
    static constexpr float zero = 0.0f;
    static constexpr float one = 1.0f;
    static constexpr float nan_value = std::numeric_limits<float>::quiet_NaN();

    template<typename T>
    static float convertToFeatureType(T inputType) {
        static_assert(std::is_floating_point_v<T>);
        return float(inputType);
    }

    static bool is_finite(float v) {
        return std::isfinite(v);
    }

    using MathType = float;
};
//! Explicit instantiation of `FeatureValues` for `double` (FP64)
template<> struct FeatureValues<double> {
    static constexpr double zero = 0.0;
    static constexpr double one = 1.0;
    static constexpr double nan_value = std::numeric_limits<double>::quiet_NaN();

    template<typename T>
    static double convertToFeatureType(T inputType) {
        static_assert(std::is_floating_point_v<T>);
        return double(inputType);
    }

    static constexpr bool is_finite(double v) {
        return std::isfinite(v);
    }

    using MathType = double;
};

//! Handling for NaN or infinite values in an array, `data`,  of `n` values.
//! @see MaskNaNStyle
template<typename T>
constexpr int64_t mask_nans(T* data, size_t n, MaskNaNStyle style, T value) {
    if (style == MaskNaNStyle::NONE) {
        return 0;
    }
    if (style == MaskNaNStyle::REPLACE) {
        for (size_t i = 0; i < n; ++i) {
            if (!FeatureValues<T>::is_finite(data[i])) {
                data[i] = value;
            }
        }
        return 0;
    }

    assert(mask_nan_style == MaskNaNStyle::REPORT);
    int64_t num_nans = 0;
    for (size_t i = 0; i < n; ++i) {
        num_nans += (!FeatureValues<T>::is_finite(data[i]));
    }
    return num_nans;
}


// This is just a function to provide to torch, so that we don't have to copy
// the tensor data to put it in a torch tensor, and torch can delete the data
// when it's no longer needed.
template<typename T>
void deleter(void* p) {
    delete[](T*)p;
}

//! Helper function to construct a torch `Tensor` from a C++ array.
//! The `Tensor` takes ownership of the memory owned by `source`.
template<typename T>
at::Tensor torch_tensor_from_array(std::unique_ptr<T[]>&& source, const int64_t* dims, size_t num_dims, c10::ScalarType type) {
    return at::from_blob(
        source.release(),
        at::IntArrayRef(dims, num_dims),
        deleter<T>, c10::TensorOptions(type));
}

//! Most of the data needed about an atom
struct CompactAtom {
    uint8_t atomicNum;
    uint8_t totalDegree;
    int8_t formalCharge;
    uint8_t chiralTag;
    uint8_t totalNumHs;
    uint8_t hybridization;
    bool isAromatic;
    float mass;
};

//! Most of the data needed about a bond
struct CompactBond {
    uint8_t bondType;
    bool isConjugated;
    bool isInRing;
    uint8_t stereo;
    uint32_t beginAtomIdx;
    uint32_t endAtomIdx;
};

//! Data representing a molecule before featurization
struct GraphData {
    const size_t num_atoms;
    std::unique_ptr<CompactAtom[]> atoms;
    const size_t num_bonds;
    std::unique_ptr<CompactBond[]> bonds;

    std::unique_ptr<RDKit::RWMol> mol;
};


//! This is called from Python to list atom one-hot features in a format that will be faster
//! to interpret inside `featurize_smiles`, passed in the `atom_property_list_onehot` parameter.
//! Implemented in features.cpp, but declared here so that graphium_cpp.cpp can expose them to
//! Python via pybind.
at::Tensor atom_onehot_feature_names_to_tensor(const std::vector<std::string>& features);

//! This is called from Python to list atom float features in a format that will be faster
//! to interpret inside `featurize_smiles`, passed in the `atom_property_list_float` parameter.
//! Implemented in features.cpp, but declared here so that graphium_cpp.cpp can expose them to
//! Python via pybind.
at::Tensor atom_float_feature_names_to_tensor(const std::vector<std::string>& features);

//! This is called from Python to list bond features in a format that will be faster
//! to interpret inside `featurize_smiles`, passed in the `bond_property_list` parameter.
//! Implemented in features.cpp, but declared here so that graphium_cpp.cpp can expose them to
//! Python via pybind.
at::Tensor bond_feature_names_to_tensor(const std::vector<std::string>& features);

//! This is called from Python to list positional features and their options in a format that
//! will be faster to interpret inside `featurize_smiles`, passed in the `bond_property_list`
//! parameter.  Implemented in features.cpp, but declared here so that graphium_cpp.cpp can
//! expose them to Python via pybind.
std::pair<std::vector<std::string>,at::Tensor> positional_feature_options_to_tensor(const pybind11::dict& dict);

//! `featurize_smiles` is called from Python to get feature tensors for `smiles_string`.
//!
//! @param smiles_string SMILES string of the molecule to featurize
//! @param atom_property_list_onehot Torch `Tensor` returned by
//!                                  `atom_onehot_feature_names_to_tensor` representing the
//!                                  list of one-hot atom features to create.
//! @param atom_property_list_float Torch `Tensor` returned by
//!                                 `atom_float_feature_names_to_tensor` representing the
//!                                 list of float atom features to create.
//! @param create_conformer_feature If true, a feature `Tensor` for a conformer is created.
//! @param bond_property_list Torch `Tensor` returned by `bond_feature_names_to_tensor`
//!                           representing the list of bond features to create.
//! @param positional_property_list Torch `Tensor` returned by
//!                                 `positional_feature_options_to_tensor` representing the list
//!                                 of positional features to create and their options.
//! @param duplicate_edges If true (the default), bond features will have values stored for
//!                        both edge directions.
//! @param add_self_loop If true (default false), bond features will have values stored for
//!                      self-edges.
//! @param explicit_H If true (default false), implicit hydrogen atoms will be added explicitly
//!                   before featurizing.
//! @param use_bonds_weights If true (default false), some features may use the bond type as an
//!                          edge weight, e.g. 2.0 for double bonds or 1.5 for aromatic bonds.
//! @param offset_carbon If true (the default), some atom float features will subtract a
//!                      value representing carbon, so that carbon atoms would have value zero.
//! @param dtype_int Value representing the torch data type to use for the output `Tensor`s.
//!                  Allowed values are 5 (FP16), 6 (FP32), and 7 (FP64), corresponding with
//!                  `c10::ScalarType`.
//! @param mask_nan_style_int Value representing the behaviour for handling NaN and infinite
//!                           output values.  Allowed values are 0 (ignore NaNs), 1 (return
//!                           the number of NaNs and the index of the first output `Tensor`
//!                           containing NaNs), and 2 (replace NaN values with `mask_nan_value`)
//!                           corresponding with the `MaskNaNStyle` enum.
//! @param mask_nan_value Value to replace NaN and infinite values with if `mask_nan_style_int`
//!                       is 2 (`MaskNaNStyle::REPLACE`)
//! @return A vector of torch `Tensor`s for the features, as well as two integers representing
//!         the number of NaN values and the index of the first output `Tensor` containing NaNs
//!         if `mask_nan_style_int` is 1 (`MaskNaNStyle::REPORT`).  The first tensor is a 2 by
//!         `num_edges` (taking into account `duplicate_edges` and `add_self_loop`) 64-bit
//!         integer `Tensor` with the atom indices on either side of each edge.  The second
//!         tensor is a 1D `Tensor` with length `num_edges`, containing all ones, even if
//!         `use_bonds_weights` is true.  The third tensor is the atom features tensor,
//!         `num_atoms` by the number of values required for all one-hot and float atom
//!         features.  The fourth tensor is the bond features tensor, `num_edges` by the number
//!         of values required for all bond features.  If `create_conformer_feature` is true,
//!         the fifth tensor is a 1D tensor of length `3*num_atoms` for the conformer positions.
//!         The rest of the tensors are the positional feature tensors, one for each positional
//!         feature.
std::tuple<std::vector<at::Tensor>, int64_t, int64_t> featurize_smiles(
    const std::string& smiles_string,
    const at::Tensor& atom_property_list_onehot,
    const at::Tensor& atom_property_list_float,
    bool create_conformer_feature,
    const at::Tensor& bond_property_list,
    const at::Tensor& positional_property_list,
    bool duplicate_edges = true,
    bool add_self_loop = false,
    bool explicit_H = false,
    bool use_bonds_weights = false,
    bool offset_carbon = true,
    int dtype_int = int(c10::ScalarType::Half),
    int mask_nan_style_int = int(MaskNaNStyle::REPORT),
    double mask_nan_value = 0.0);

//! Creates an RWMol from a SMILES string.
//!
//! If `ordered` is true, and the string contains atom classes, called "bookmarks" in RDKit,
//! that form a complete (0-based) ordering of the atoms, the atoms will be reordered according
//! to this explicit order, and the bookmarks will be removed, so that canonical orders
//! can be correctly compared later.
//!
//! This is implemented in graphium_cpp.cpp, but is declared in this header so
//! that both labels.cpp and features.cpp can call it.
std::unique_ptr<RDKit::RWMol> parse_mol(
    const std::string& smiles_string,
    bool explicit_H,
    bool ordered = true);

//! Determines a canonical ordering of the atoms in `mol`
//!
//! This is implemented in graphium_cpp.cpp, to keep it near `parse_mol`
void get_canonical_atom_order(
    const RDKit::ROMol& mol,
    std::vector<unsigned int>& atom_order);
