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

enum class FeatureLevel {
    NODE,
    EDGE,
    NODEPAIR,
    GRAPH
};

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

enum class AtomOneHotFeature {
    ATOMIC_NUM,
    DEGREE,
    VALENCE,
    IMPLICIT_VALENCE,
    HYBRIDIZATION,
    CHIRALITY,
    PHASE,
    TYPE,
    GROUP,
    PERIOD,
    UNKNOWN
};

enum class BondFeature {
    TYPE_FLOAT,
    TYPE_ONE_HOT,
    IN_RING,
    CONJUGATED,
    STEREO_ONE_HOT,
    CONFORMER_BOND_LENGTH,
    ESTIMATED_BOND_LENGTH,
    UNKNOWN
};

enum class PositionalFeature {
    LAPLACIAN_EIGENVEC,
    LAPLACIAN_EIGENVAL,
    RW_RETURN_PROBS,
    RW_TRANSITION_PROBS,
    ELECTROSTATIC,
    COMMUTE,
    GRAPHORMER
};

enum class Normalization {
    NONE,
    SYMMETRIC,
    INVERSE
};

enum class MaskNaNStyle {
    NONE,
    REPORT,
    REPLACE
};

struct PositionalOptions {
    PositionalFeature feature;
    FeatureLevel level;

    std::vector<uint32_t> rw_powers;
    int rw_space_dim = 0;

    uint32_t laplacian_num_pos = 8;
    Normalization laplacian_normalization = Normalization::NONE;
    bool laplacian_disconnected_comp = true;
};

template<typename T>
struct FeatureValues {};

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

template<typename T>
at::Tensor torch_tensor_from_array(std::unique_ptr<T[]>&& source, const int64_t* dims, size_t num_dims, c10::ScalarType type) {
    return at::from_blob(
        source.release(),
        at::IntArrayRef(dims, num_dims),
        deleter<T>, c10::TensorOptions(type));
}

// Most of the data needed about an atom
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

// Most of the data needed about a bond
struct CompactBond {
    uint8_t bondType;
    bool isConjugated;
    bool isInRing;
    uint8_t stereo;
    uint32_t beginAtomIdx;
    uint32_t endAtomIdx;
};

// Data representing a molecule before featurization
struct GraphData {
    const size_t num_atoms;
    std::unique_ptr<CompactAtom[]> atoms;
    const size_t num_bonds;
    std::unique_ptr<CompactBond[]> bonds;

    std::unique_ptr<RDKit::RWMol> mol;
};


// These functions are in features.cpp, and declared here so that
// graphium_cpp.cpp can expose them to Python via pybind.
at::Tensor atom_float_feature_names_to_tensor(const std::vector<std::string>& features);
at::Tensor atom_onehot_feature_names_to_tensor(const std::vector<std::string>& features);
at::Tensor bond_feature_names_to_tensor(const std::vector<std::string>& features);
std::pair<std::vector<std::string>,at::Tensor> positional_feature_options_to_tensor(const pybind11::dict& dict);
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


// parse_mol is in graphium_cpp.cpp, but is declared in this header so
// that both labels.cpp and features.cpp can call it.
std::unique_ptr<RDKit::RWMol> parse_mol(
    const std::string& smiles_string,
    bool explicit_H,
    bool ordered = false);

// Determines a canonical ordering of the atoms
void get_canonical_atom_order(
    const RDKit::ROMol& mol,
    std::vector<unsigned int>& atom_order);
