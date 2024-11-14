// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
@file
*/


#include "one_hot.h"
#include "features.h"
#include "float_features.h"

#include <GraphMol/ROMol.h>
#include <RDGeneral/types.h>

#include <stdint.h>
#include <string>
#include <string.h>
#include <type_traits>

// Helper class to automatically generates a reverse lookup table at compile time,
// with `MAX_OUT` used as a sentinel to indicate that a value wasn't present
// in the original list.
template<size_t NUM_IN, size_t MAX_OUT>
class OneHotLookup {
    size_t indices[NUM_IN];
public:
    constexpr OneHotLookup(const size_t list[MAX_OUT]) : indices() {
        std::fill(indices, indices + NUM_IN, MAX_OUT);
        for (size_t i = 0; i < MAX_OUT; ++i) {
            indices[list[i]] = i;
        }
    }
    constexpr size_t operator[](size_t i) const {
        return (i < NUM_IN) ? indices[i] : MAX_OUT;
    }
};

// This list of elements matches ATOM_LIST in older file graphium/features/nmp.py
constexpr size_t atomicNumList[] = {
    6 -1, // C
    7 -1, // N
    8 -1, // O
    16-1,// S
    9 -1, // F
    14-1,// Si
    15-1,// P
    17-1,// Cl
    35-1,// Br
    12-1,// Mg
    11-1,// Na
    20-1,// Ca
    26-1,// Fe
    33-1,// As
    13-1,// Al
    53-1,// I
    5 -1,// B
    23-1,// V
    19-1,// K
    81-1,// Tl
    70-1,// Yb
    51-1,// Sb
    50-1,// Sn
    47-1,// Ag
    46-1,// Pd
    27-1,// Co
    34-1,// Se
    22-1,// Ti
    30-1,// Zn
    1 -1,// H
    3 -1,// Li
    32-1,// Ge
    29-1,// Cu
    79-1,// Au
    28-1,// Ni
    48-1,// Cd
    49-1,// In
    25-1,// Mn
    40-1,// Zr
    24-1,// Cr
    78-1,// Pt
    80-1,// Hg
    82-1,// Pb
};
constexpr size_t atomicNumCount = std::extent<decltype(atomicNumList)>::value;
constexpr OneHotLookup<118, atomicNumCount> atomicNumLookup(atomicNumList);

constexpr size_t degreeCount = 5;
constexpr size_t valenceCount = 7;

// Reverse alphabetical order, excluding "OTHER",
// matching HYBRIDIZATION_LIST in older file graphium/features/nmp.py
constexpr size_t hybridizationList[] = {
    RDKit::Atom::HybridizationType::UNSPECIFIED,
    RDKit::Atom::HybridizationType::SP3D2,
    RDKit::Atom::HybridizationType::SP3D,
    RDKit::Atom::HybridizationType::SP3,
    RDKit::Atom::HybridizationType::SP2D,
    RDKit::Atom::HybridizationType::SP2,
    RDKit::Atom::HybridizationType::SP,
    RDKit::Atom::HybridizationType::S,
};
constexpr size_t hybridizationCount = std::extent<decltype(hybridizationList)>::value;
constexpr OneHotLookup<8, hybridizationCount> hybridizationLookup(hybridizationList);

static const std::string chiralityRString("R");

enum ElementPhase {
    GAS,
    ARTIFICIAL,
    LIQ,
    SOLID
};
// This table is from the Phase column of graphium/features/periodic_table.csv
constexpr ElementPhase atomicNumToPhase[] = {
    GAS, GAS,
    SOLID, SOLID, SOLID, SOLID, GAS, GAS, GAS, GAS,
    SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, GAS, GAS,
    SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, LIQ, GAS,
    SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, ARTIFICIAL, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, GAS,
    SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, ARTIFICIAL, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, LIQ, SOLID, SOLID, SOLID, SOLID, SOLID, GAS,
    SOLID, SOLID, SOLID, SOLID, SOLID, SOLID, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL, ARTIFICIAL,
};
constexpr size_t phaseCount = 4;

enum ElementType {
    NOBLE_GAS,
    ALKALI_METAL,
    METAL, HALOGEN,
    LANTHANIDE,
    ALKALINE_EARTH_METAL,
    TRANSITION_METAL,
    ACTINIDE,
    METALLOID,
    NONE,
    TRANSACTINIDE,
    NONMETAL,

    NUM_ELEMENT_TYPES
};
// This table is from the Type column of graphium/features/periodic_table.csv
constexpr ElementType atomicNumToType[] = {
    NONMETAL, NOBLE_GAS,
    ALKALI_METAL, ALKALINE_EARTH_METAL, METALLOID, NONMETAL, NONMETAL, NONMETAL, HALOGEN, NOBLE_GAS,
    ALKALI_METAL, ALKALINE_EARTH_METAL, METAL, METALLOID, NONMETAL, NONMETAL, HALOGEN, NOBLE_GAS,
    ALKALI_METAL, ALKALINE_EARTH_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, METAL, METALLOID, METALLOID, NONMETAL, HALOGEN, NOBLE_GAS,
    ALKALI_METAL, ALKALINE_EARTH_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, METAL, METAL, METALLOID, METALLOID, HALOGEN, NOBLE_GAS,
    ALKALI_METAL, ALKALINE_EARTH_METAL, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, LANTHANIDE, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, TRANSITION_METAL, METAL, METAL, METAL, METALLOID, NOBLE_GAS,
    ALKALI_METAL, ALKALINE_EARTH_METAL, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, ACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, TRANSACTINIDE, NONE, TRANSACTINIDE, NONE, TRANSACTINIDE, NONE, NOBLE_GAS
};
constexpr size_t typeCount = ElementType::NUM_ELEMENT_TYPES;

// This matches BOND_TYPES in older file graphium/features/nmp.py
constexpr size_t bondTypeList[] = {
    RDKit::Bond::BondType::SINGLE,
    RDKit::Bond::BondType::DOUBLE,
    RDKit::Bond::BondType::TRIPLE,
    RDKit::Bond::BondType::AROMATIC,
};
constexpr size_t bondTypeCount = std::extent<decltype(bondTypeList)>::value;
constexpr OneHotLookup<22, bondTypeCount> bondTypeLookup(bondTypeList);

// This matches BOND_STEREO in older file graphium/features/nmp.py
constexpr size_t bondStereoList[] = {
    RDKit::Bond::BondStereo::STEREONONE,
    RDKit::Bond::BondStereo::STEREOANY,
    RDKit::Bond::BondStereo::STEREOZ,
    RDKit::Bond::BondStereo::STEREOE,
    RDKit::Bond::BondStereo::STEREOCIS,
    RDKit::Bond::BondStereo::STEREOTRANS,
};
constexpr size_t bondStereoCount = std::extent<decltype(bondStereoList)>::value;
constexpr OneHotLookup<6, bondStereoCount> bondStereoLookup(bondStereoList);

// Returns the number of values per atom, required by `feature` in `get_one_hot_atom_feature`'s
// `data` argument.
size_t get_one_hot_atom_feature_size(AtomOneHotFeature feature) {
    switch (feature) {
    case AtomOneHotFeature::ATOMIC_NUM:       return atomicNumCount + 1;
    case AtomOneHotFeature::DEGREE:           return degreeCount + 1;
    case AtomOneHotFeature::VALENCE:          return valenceCount + 1;
    case AtomOneHotFeature::IMPLICIT_VALENCE: return valenceCount + 1;
    case AtomOneHotFeature::HYBRIDIZATION:    return hybridizationCount + 1;
    // "R", anything else ("S" or no value), bool for if other property present
    case AtomOneHotFeature::CHIRALITY:        return 3;
    case AtomOneHotFeature::PHASE:            return phaseCount + 1;
    case AtomOneHotFeature::TYPE:             return typeCount + 1;
    case AtomOneHotFeature::GROUP:            return groupCount + 1;
    case AtomOneHotFeature::PERIOD:           return periodCount + 1;
    default:
        // Missing implementation
        assert(0);
        return 0;
    }
}

// Fills in a particular atom `feature`'s one-hot encoding into `data`, for all atoms.
// See the declaration in one_hot.h for more details.
template<typename T>
size_t get_one_hot_atom_feature(const GraphData& graph, T* data, AtomOneHotFeature feature, size_t stride) {
    const size_t num_atoms = graph.num_atoms;
    const RDKit::ROMol& mol = *graph.mol.get();
    const size_t feature_size = get_one_hot_atom_feature_size(feature);
    const size_t total_feature_size = feature_size * num_atoms;
    if (total_feature_size == 0) {
        return feature_size;
    }
    {
        T* current_data = data;
        for (size_t i = 0; i < num_atoms; ++i) {
            memset(current_data, 0, sizeof(data[0]) * feature_size);
            current_data += stride;
        }
    }
    switch (feature) {
    case AtomOneHotFeature::ATOMIC_NUM:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            size_t atomicNum = graph.atoms[atomIndex].atomicNum;
            data[atomicNumLookup[atomicNum-1]] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::DEGREE:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            auto degree = mol.getAtomWithIdx(atomIndex)->getDegree();
            size_t dataIndex = (degree < degreeCount) ? degree : degreeCount;
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::VALENCE:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            auto valence = mol.getAtomWithIdx(atomIndex)->getTotalValence();
            size_t dataIndex = (size_t(valence) < valenceCount) ? size_t(valence) : valenceCount;
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::IMPLICIT_VALENCE:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            auto valence = mol.getAtomWithIdx(atomIndex)->getImplicitValence();
            size_t dataIndex = (size_t(valence) < valenceCount) ? size_t(valence) : valenceCount;
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::HYBRIDIZATION:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            auto hybridization = mol.getAtomWithIdx(atomIndex)->getHybridization();
            data[hybridizationLookup[hybridization]] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::CHIRALITY:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            std::string chirality;
            const RDKit::Atom* atom = mol.getAtomWithIdx(atomIndex);
            bool isPresent = atom->getPropIfPresent(RDKit::common_properties::_CIPCode, chirality);
            data[(isPresent && chirality == chiralityRString) ? 0 : 1] = FeatureValues<T>::one;
            if (atom->hasProp(RDKit::common_properties::_ChiralityPossible)) {
                data[2] = FeatureValues<T>::one;
            }
        }
        return feature_size;
    case AtomOneHotFeature::PHASE:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            size_t atomicNum = graph.atoms[atomIndex].atomicNum;
            size_t dataIndex = phaseCount;
            if (atomicNum - 1 < std::extent<decltype(atomicNumToPhase)>::value) {
                ElementPhase phase = atomicNumToPhase[atomicNum - 1];
                // Group numbers are 1-based, but the array indices aren't.
                dataIndex = phase - 1;
            }
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::TYPE:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            size_t atomicNum = graph.atoms[atomIndex].atomicNum;
            size_t dataIndex = typeCount;
            if (atomicNum - 1 < std::extent<decltype(atomicNumToType)>::value) {
                ElementType type = atomicNumToType[atomicNum - 1];
                // Group numbers are 1-based, but the array indices aren't.
                dataIndex = type - 1;
            }
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::GROUP:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            size_t atomicNum = graph.atoms[atomIndex].atomicNum;
            size_t dataIndex = groupCount;
            if (atomicNum - 1 < std::extent<decltype(atomicNumToGroupTable)>::value) {
                uint8_t group = atomicNumToGroupTable[atomicNum - 1];
                // Group numbers are 1-based, but the array indices aren't.
                dataIndex = group - 1;
            }
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    case AtomOneHotFeature::PERIOD:
        for (size_t atomIndex = 0; atomIndex < num_atoms; ++atomIndex, data += stride) {
            size_t atomicNum = graph.atoms[atomIndex].atomicNum;
            size_t dataIndex = periodCount;
            if (atomicNum - 1 < std::extent<decltype(atomicNumToPeriodTable)>::value) {
                uint8_t period = atomicNumToPeriodTable[atomicNum - 1];
                // Period numbers are 1-based, but the array indices aren't.
                dataIndex = period - 1;
            }
            data[dataIndex] = FeatureValues<T>::one;
        }
        return feature_size;
    default:
        // Missing implementation
        assert(0);
        return feature_size;
    }
}

// Explicit instantiations, so that the function can be templated
// but still be used from other cpp files.
template size_t get_one_hot_atom_feature<int16_t>(const GraphData& graph, int16_t* data, AtomOneHotFeature feature, size_t stride);
template size_t get_one_hot_atom_feature<float>(const GraphData& graph, float* data, AtomOneHotFeature feature, size_t stride);
template size_t get_one_hot_atom_feature<double>(const GraphData& graph, double* data, AtomOneHotFeature feature, size_t stride);


// Returns the number of values per bond, required by `feature` in `get_one_hot_bond_feature`'s
// `data` argument.
size_t get_one_hot_bond_feature_size(BondFeature feature) {
    switch (feature) {
    case BondFeature::TYPE_ONE_HOT:   return bondTypeCount + 1;
    case BondFeature::STEREO_ONE_HOT: return bondStereoCount + 1;
    default:
        break;
    }
    // Missing implementation
    assert(0);
    return 0;
}

// Fills in a particular bond `feature`'s one-hot encoding into `data`, for all bonds.
// See the declaration in one_hot.h for more details.
template<typename T>
size_t get_one_hot_bond_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride) {
    const size_t num_bonds = graph.num_bonds;
    const size_t feature_size = get_one_hot_bond_feature_size(feature);
    const size_t total_feature_size = feature_size * num_bonds;
    if (total_feature_size == 0) {
        return 0;
    }
    {
        T* current_data = data;
        for (size_t i = 0; i < num_bonds; ++i) {
            memset(current_data, 0, sizeof(data[0]) * feature_size);
            current_data += stride;
        }
    }
    switch (feature) {
    case BondFeature::TYPE_ONE_HOT:
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            auto type = graph.bonds[i].bondType;
            data[bondTypeLookup[type]] = FeatureValues<T>::one;
        }
        return feature_size;
    case BondFeature::STEREO_ONE_HOT:
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            auto stereo = graph.bonds[i].stereo;
            data[bondStereoLookup[stereo]] = FeatureValues<T>::one;
        }
        return feature_size;
    default:
        // Missing implementation
        assert(0);
        return feature_size;
    }
}

// Explicit instantiations, so that the function can be templated
// but still be used from other cpp files.
template size_t get_one_hot_bond_feature<int16_t>(const GraphData& graph, int16_t* data, BondFeature feature, size_t stride);
template size_t get_one_hot_bond_feature<float>(const GraphData& graph, float* data, BondFeature feature, size_t stride);
template size_t get_one_hot_bond_feature<double>(const GraphData& graph, double* data, BondFeature feature, size_t stride);
