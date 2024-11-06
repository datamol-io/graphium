// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
@file
*/


#include "float_features.h"

#include "features.h"

#include <GraphMol/Atom.h>
#include <GraphMol/Bond.h>
#include <GraphMol/PeriodicTable.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <RDGeneral/types.h>

#include <stdint.h>
#include <cmath>

static constexpr double qNaN = std::numeric_limits<double>::quiet_NaN();

// This table is from the Electronegativity column of graphium/features/periodic_table.csv
const double electronegativityTable[] = {
          2.20, qNaN, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98,
    qNaN, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, qNaN, 0.82,
    1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90,
    1.65, 1.81, 2.01, 2.18, 2.55, 2.96, qNaN, 0.82, 0.95, 1.22,
    1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69, 1.78,
    1.96, 2.05, 2.10, 2.66, qNaN, 0.79, 0.89, 1.10, 1.12, 1.13,
    1.14, 1.13, 1.17, 1.20, 1.20, 1.20, 1.22, 1.23, 1.24, 1.25,
    1.10, 1.27, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54,
    2.00, 2.04, 2.33, 2.02, 2.00, 2.20, qNaN, 0.70, 0.90, 1.10,
    1.30, 1.50, 1.38, 1.36, 1.28, 1.30, 1.30, 1.30, 1.30, 1.30,
    1.30, 1.30, 1.30, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN,
    qNaN, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN,
};

// This table is from the FirstIonization column of graphium/features/periodic_table.csv
const double firstIonizationTable[] = {
             13.5984, 24.5874,  5.3917,  9.3227,  8.2980, 11.2603, 14.5341, 13.6181, 17.4228,
    21.5645,  5.1391,  7.6462,  5.9858,  8.1517, 10.4867, 10.3600, 12.9676, 15.7596,  4.3407,
     6.1132,  6.5615,  6.8281,  6.7462,  6.7665,  7.4340,  7.9024,  7.8810,  7.6398,  7.7264,
     9.3942,  5.9993,  7.8994,  9.7886,  9.7524, 11.8138, 13.9996,  4.1771,  5.6949,  6.2173,
     6.6339,  6.7589,  7.0924,  7.2800,  7.3605,  7.4589,  8.3369,  7.5762,  8.9938,  5.7864,
     7.3439,  8.6084,  9.0096, 10.4513, 12.1298,  3.8939,  5.2117,  5.5769,  5.5387,  5.4730,
     5.5250,  5.5820,  5.6437,  5.6704,  6.1501,  5.8638,  5.9389,  6.0215,  6.1077,  6.1843,
     6.2542,  5.4259,  6.8251,  7.5496,  7.8640,  7.8335,  8.4382,  8.9670,  8.9587,  9.2255,
    10.4375,  6.1082,  7.4167,  7.2856,  8.4170,  9.3000, 10.7485,  4.0727,  5.2784,  5.1700,
     6.3067,  5.8900,  6.1941,  6.2657,  6.0262,  5.9738,  5.9915,  6.1979,  6.2817,  6.4200,
     6.5000,  6.5800,  6.6500,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,
      qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,
};

// This table is from the MeltingPoint column of graphium/features/periodic_table.csv
const double meltingPointTable[] = {
              14.175,   qNaN ,  453.85, 1560.15, 2573.15, 3948.15,   63.29,   50.50,   53.63,
     24.703,  371.15,  923.15,  933.40, 1683.15,  317.25,  388.51,  172.31,   83.96,  336.50,
    1112.15, 1812.15, 1933.15, 2175.15, 2130.15, 1519.15, 1808.15, 1768.15, 1726.15, 1357.75,
     692.88,  302.91, 1211.45, 1090.15,  494.15,  266.05,  115.93,  312.79, 1042.15, 1799.15,
    2125.15, 2741.15, 2890.15, 2473.15, 2523.15, 2239.15, 1825.15, 1234.15,  594.33,  429.91,
     505.21,  904.05,  722.80,  386.65,  161.45,  301.70, 1002.15, 1193.15, 1071.15, 1204.15,
    1289.15, 1204.15, 1345.15, 1095.15, 1585.15, 1630.15, 1680.15, 1743.15, 1795.15, 1818.15,
    1097.15, 1936.15, 2500.15, 3269.15, 3680.15, 3453.15, 3300.15, 2716.15, 2045.15, 1337.73,
     234.43,  577.15,  600.75,  544.67,  527.15,  575.15,  202.15,  300.15,  973.15, 1323.15,
    2028.15, 1873.15, 1405.15,  913.15,  913.15, 1267.15, 1340.15, 1259.15, 1925.15, 1133.15,
      qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,
      qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,   qNaN ,
};

// This table is 2x the Metal column plus the Metalloid column of graphium/features/periodic_table.csv
const uint8_t metalTable[] = {
       0, 0, 2, 2, 1, 0, 0, 0, 0,
    0, 2, 2, 2, 1, 0, 0, 0, 0, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 1, 1, 0, 0, 0, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 1, 1, 0, 0, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 1, 0, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 0, 0,
};

// Fills in a particular atom float `feature` into `data`, for all atoms.
// See the declaration in float_features.h for more details.
template<typename T>
void get_atom_float_feature(const GraphData& graph, T* data, AtomFloatFeature feature, size_t stride, bool offset_carbon) {
    const uint32_t num_atoms = graph.num_atoms;
    constexpr uint32_t carbon_atomic_num = 6;
    using MT = typename FeatureValues<T>::MathType;
    switch (feature) {
    case AtomFloatFeature::ATOMIC_NUMBER: {
        const MT offset = offset_carbon ? carbon_atomic_num : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType((MT(graph.atoms[i].atomicNum) - offset) / MT(5));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::MASS: {
        const RDKit::ROMol& mol = *graph.mol.get();
        constexpr MT carbon_mass = MT(12.011);
        const MT offset = offset_carbon ? carbon_mass : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType((MT(mol.getAtomWithIdx(i)->getMass()) - offset) / MT(10));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::VALENCE: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const MT offset = offset_carbon ? 4 : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getTotalValence()) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::IMPLICIT_VALENCE: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getImplicitValence()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::HYBRIDIZATION: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getHybridization()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::CHIRALITY: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const RDKit::Atom* atom = mol.getAtomWithIdx(i);
            std::string prop;
            bool has_prop = atom->getPropIfPresent(RDKit::common_properties::_CIPCode, prop);
            *data = FeatureValues<T>::convertToFeatureType(has_prop ? MT(prop.length() == 1 && prop[0] == 'R') : MT(2));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::AROMATIC: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getIsAromatic()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::IN_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(ring_info->numAtomRings(i) != 0));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::MIN_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(ring_info->minAtomRingSize(i)));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::MAX_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            data[i * stride] = FeatureValues<T>::zero;
        }
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        const auto& rings = ring_info->atomRings();
        for (const auto& ring : rings) {
            const T size = FeatureValues<T>::convertToFeatureType(MT(ring.size()));
            for (const auto atom_index : ring) {
                if (size > data[atom_index * stride]) {
                    data[atom_index * stride] = size;
                }
            }
        }
        return;
    }
    case AtomFloatFeature::NUM_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::RingInfo* ring_info = mol.getRingInfo();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(ring_info->numAtomRings(i)));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::DEGREE: {
        const RDKit::ROMol& mol = *graph.mol.get();
        const MT offset = offset_carbon ? 2 : 0;
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getTotalDegree()) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::RADICAL_ELECTRON: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(mol.getAtomWithIdx(i)->getNumRadicalElectrons()));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::FORMAL_CHARGE: {
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(graph.atoms[i].formalCharge));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::VDW_RADIUS: {
        const RDKit::PeriodicTable* table = RDKit::PeriodicTable::getTable();
        const MT offset = offset_carbon ? MT(table->getRvdw(carbon_atomic_num)) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(table->getRvdw(graph.atoms[i].atomicNum)) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::COVALENT_RADIUS: {
        const RDKit::PeriodicTable* table = RDKit::PeriodicTable::getTable();
        const MT offset = offset_carbon ? MT(table->getRcovalent(carbon_atomic_num)) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(table->getRcovalent(graph.atoms[i].atomicNum)) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::ELECTRONEGATIVITY: {
        const MT offset = offset_carbon ? MT(electronegativityTable[carbon_atomic_num-1]) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i, data += stride) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            if (atomic_num <= 0 || atomic_num > 118 || electronegativityTable[atomic_num - 1] == 0) {
                *data = FeatureValues<T>::nan_value;
                continue;
            }
            *data = FeatureValues<T>::convertToFeatureType(MT(electronegativityTable[atomic_num - 1]) - offset);
        }
        return;
    }
    case AtomFloatFeature::IONIZATION: {
        const T offset = offset_carbon ? T(firstIonizationTable[carbon_atomic_num-1]) : T(0);
        for (uint32_t i = 0; i < num_atoms; ++i, data += stride) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            if (atomic_num <= 0 || atomic_num > 118 || firstIonizationTable[atomic_num - 1] == 0) {
                *data = FeatureValues<T>::nan_value;
                continue;
            }
            *data = FeatureValues<T>::convertToFeatureType((MT(firstIonizationTable[atomic_num - 1]) - offset) / MT(5));
        }
        return;
    }
    case AtomFloatFeature::MELTING_POINT: {
        const MT offset = offset_carbon ? MT(meltingPointTable[carbon_atomic_num-1]) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i, data += stride) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            if (atomic_num <= 0 || atomic_num > 118 || meltingPointTable[atomic_num - 1] == 0) {
                *data = FeatureValues<T>::nan_value;
                continue;
            }
            *data = FeatureValues<T>::convertToFeatureType((MT(meltingPointTable[atomic_num - 1]) - offset) / MT(200));
        }
        return;
    }
    case AtomFloatFeature::METAL: {
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            *data = (atomic_num <= 0 || atomic_num > 118) ? FeatureValues<T>::nan_value : FeatureValues<T>::convertToFeatureType(MT(metalTable[atomic_num - 1]));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::GROUP: {
        const MT offset = offset_carbon ? MT(atomicNumToGroupTable[carbon_atomic_num - 1]) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            *data = (atomic_num <= 0 || atomic_num > 118) ? FeatureValues<T>::nan_value : FeatureValues<T>::convertToFeatureType(MT(atomicNumToGroupTable[atomic_num - 1]) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::PERIOD: {
        const MT offset = offset_carbon ? MT(atomicNumToPeriodTable[carbon_atomic_num - 1]) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            const uint32_t atomic_num = graph.atoms[i].atomicNum;
            *data = (atomic_num <= 0 || atomic_num > 118) ? FeatureValues<T>::nan_value : FeatureValues<T>::convertToFeatureType(MT(atomicNumToPeriodTable[atomic_num - 1]) - offset);
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::SINGLE_BOND:
    case AtomFloatFeature::AROMATIC_BOND:
    case AtomFloatFeature::DOUBLE_BOND:
    case AtomFloatFeature::TRIPLE_BOND:
    {
        const RDKit::ROMol& mol = *graph.mol.get();
        const RDKit::Bond::BondType type =
            (feature == AtomFloatFeature::SINGLE_BOND) ? RDKit::Bond::SINGLE : (
                (feature == AtomFloatFeature::AROMATIC_BOND) ? RDKit::Bond::AROMATIC : (
                (feature == AtomFloatFeature::DOUBLE_BOND) ? RDKit::Bond::DOUBLE : (
                RDKit::Bond::TRIPLE)));
        for (uint32_t i = 0; i < num_atoms; ++i) {
            auto [begin, end] = mol.getAtomBonds(mol.getAtomWithIdx(i));
            uint32_t count = 0;
            for (; begin != end; ++begin) {
                count += (mol[*begin]->getBondType() == type);
            }
            *data = FeatureValues<T>::convertToFeatureType(MT(count));
            data += stride;
        }
        return;
    }
    case AtomFloatFeature::IS_CARBON: {
        const MT offset = offset_carbon ? MT(1) : MT(0);
        for (uint32_t i = 0; i < num_atoms; ++i) {
            *data = FeatureValues<T>::convertToFeatureType(MT(graph.atoms[i].atomicNum == carbon_atomic_num) - offset);
            data += stride;
        }
        return;
    }
    default:
        break;
    }

    // Missing implementation
    assert(0);
    for (uint32_t i = 0; i < num_atoms; ++i) {
        *data = FeatureValues<T>::nan_value;
        data += stride;
    }
}

// Explicit instantiations, so that the function can be templated
// but still be used from other cpp files.
template void get_atom_float_feature<int16_t>(const GraphData& graph, int16_t* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
template void get_atom_float_feature<float>(const GraphData& graph, float* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);
template void get_atom_float_feature<double>(const GraphData& graph, double* data, AtomFloatFeature feature, size_t stride, bool offset_carbon);

// This table is from the SingleBondRadius column of graphium/features/periodic_table.csv
const double single_bond_lengths[] = {
          0.32, 0.46, 1.33, 1.02, 0.85, 0.75, 0.71, 0.63, 0.64,
    0.67, 1.55, 1.39, 1.26, 1.16, 1.11, 1.03, 0.99, 0.96, 1.96,
    1.71, 1.48, 1.36, 1.34, 1.22, 1.19, 1.16, 1.11, 1.10, 1.12,
    1.18, 1.24, 1.21, 1.21, 1.16, 1.14, 1.17, 2.10, 1.85, 1.63,
    1.54, 1.47, 1.38, 1.28, 1.25, 1.25, 1.20, 1.28, 1.36, 1.42,
    1.40, 1.40, 1.36, 1.33, 1.31, 2.32, 1.96, 1.80, 1.63, 1.76,
    1.74, 1.73, 1.72, 1.68, 1.69, 1.68, 1.67, 1.66, 1.65, 1.64,
    1.70, 1.62, 1.52, 1.46, 1.37, 1.31, 1.29, 1.22, 1.23, 1.24,
    1.33, 1.44, 1.44, 1.51, 1.45, 1.47, 1.42, 2.23, 2.01, 1.86,
    1.75, 1.69, 1.70, 1.71, 1.72, 1.66, 1.66, 1.68, 1.68, 1.65,
    1.67, 1.73, 1.76, 1.61, 1.57, 1.49, 1.43, 1.41, 1.34, 1.29,
    1.28, 1.21, 1.22, 1.36, 1.43, 1.62, 1.75, 1.65, 1.57,
};
// This table is from the DoubleBondRadius column of graphium/features/periodic_table.csv
const double double_bond_lengths[] = {
          qNaN, qNaN, 1.24, 0.90, 0.78, 0.67, 0.60, 0.57, 0.59,
    0.96, 1.60, 1.32, 1.13, 1.07, 1.02, 0.94, 0.95, 1.07, 1.93,
    1.47, 1.16, 1.17, 1.12, 1.11, 1.05, 1.09, 1.03, 1.01, 1.15,
    1.20, 1.17, 1.11, 1.14, 1.07, 1.09, 1.21, 2.02, 1.57, 1.30,
    1.27, 1.25, 1.21, 1.20, 1.14, 1.10, 1.17, 1.39, 1.44, 1.36,
    1.30, 1.33, 1.28, 1.29, 1.35, 2.09, 1.61, 1.39, 1.37, 1.38,
    1.37, 1.35, 1.34, 1.34, 1.35, 1.35, 1.33, 1.33, 1.33, 1.31,
    1.29, 1.31, 1.28, 1.26, 1.20, 1.19, 1.16, 1.15, 1.12, 1.21,
    1.42, 1.42, 1.35, 1.41, 1.35, 1.38, 1.45, 2.18, 1.73, 1.53,
    1.43, 1.38, 1.34, 1.36, 1.35, 1.35, 1.36, 1.39, 1.40, 1.40,
    qNaN, 1.39, qNaN, 1.41, 1.40, 1.36, 1.28, 1.28, 1.25, 1.25,
    1.16, 1.16, 1.37, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN,
};
// This table is from the TripleBondRadius column of graphium/features/periodic_table.csv
const double triple_bond_lengths[] = {
          qNaN, qNaN, qNaN, 0.85, 0.73, 0.60, 0.54, 0.53, 0.53,
    qNaN, qNaN, 1.27, 1.11, 1.02, 0.94, 0.95, 0.93, 0.96, qNaN,
    1.33, 1.14, 1.08, 1.06, 1.03, 1.03, 1.02, 0.96, 1.01, 1.20,
    qNaN, 1.21, 1.14, 1.06, 1.07, 1.10, 1.08, qNaN, 1.39, 1.24,
    1.21, 1.16, 1.13, 1.10, 1.03, 1.06, 1.12, 1.37, qNaN, 1.46,
    1.32, 1.27, 1.21, 1.25, 1.22, qNaN, 1.49, 1.39, 1.31, 1.28,
    qNaN, qNaN, qNaN, qNaN, 1.32, qNaN, qNaN, qNaN, qNaN, qNaN,
    qNaN, 1.31, 1.22, 1.19, 1.15, 1.10, 1.09, 1.07, 1.10, 1.23,
    qNaN, 1.50, 1.37, 1.35, 1.29, 1.38, 1.33, qNaN, 1.59, 1.40,
    1.36, 1.29, 1.18, 1.16, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN,
    qNaN, qNaN, qNaN, qNaN, 1.31, 1.26, 1.21, 1.19, 1.18, 1.13,
    1.12, 1.18, 1.30, qNaN, qNaN, qNaN, qNaN, qNaN, qNaN,
};

// Fills in a particular bond float `feature` into `data`, for all bonds.
// See the declaration in float_features.h for more details.
template<typename T>
void get_bond_float_feature(const GraphData& graph, T* data, BondFeature feature, size_t stride) {
    const uint32_t num_bonds = graph.num_bonds;
    switch (feature) {
    case BondFeature::TYPE_FLOAT: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            auto type = graph.bonds[i].bondType;
            double value = 0;
            switch (type) {
            case RDKit::Bond::BondType::SINGLE: value = 1.0; break;
            case RDKit::Bond::BondType::DOUBLE: value = 2.0; break;
            case RDKit::Bond::BondType::TRIPLE: value = 3.0; break;
            case RDKit::Bond::BondType::AROMATIC: value = 1.5; break;
            default: value = mol.getBondWithIdx(i)->getBondTypeAsDouble();
            }
            *data = FeatureValues<T>::convertToFeatureType(value);
        }
        return;
    }
    case BondFeature::IN_RING: {
        const RDKit::ROMol& mol = *graph.mol.get();
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            bool is_in_ring = mol.getRingInfo()->numBondRings(i) != 0;
            *data = is_in_ring ? FeatureValues<T>::one : FeatureValues<T>::zero;
        }
        return;
    }
    case BondFeature::CONJUGATED: {
        for (size_t i = 0; i < num_bonds; ++i, data += stride) {
            bool is_conjugated = graph.bonds[i].isConjugated;
            *data = is_conjugated ? FeatureValues<T>::one : FeatureValues<T>::zero;
        }
        return;
    }
    case BondFeature::CONFORMER_BOND_LENGTH: {
        RDKit::ROMol& mol = *graph.mol.get();
        if (mol.beginConformers() == mol.endConformers()) {
            // Try to generate a conformer
            RDKit::DGeomHelpers::EmbedParameters params;
            params.enforceChirality = false;
            params.ignoreSmoothingFailures = true;
            params.useBasicKnowledge = true;
            params.useExpTorsionAnglePrefs = true;
            params.optimizerForceTol = 0.1;
            int id = RDKit::DGeomHelpers::EmbedMolecule(mol, params);
            if (id == -1) {
                // Failed to generate a conformer
                const uint32_t num_bonds = graph.num_bonds;
                for (uint32_t i = 0; i < num_bonds; ++i, data += stride) {
                    *data = FeatureValues<T>::nan_value;
                }
                return;
            }
            assert(mol.beginConformers() != mol.endConformers());
        }
        const RDKit::Conformer& conformer = mol.getConformer();
        const auto& positions = conformer.getPositions();
        for (uint32_t i = 0; i < num_bonds; ++i, data += stride) {
            const uint32_t begin_atom = graph.bonds[i].beginAtomIdx;
            const uint32_t end_atom = graph.bonds[i].endAtomIdx;
            const RDGeom::Point3D diff = (positions[end_atom] - positions[begin_atom]);
            // Unfortunately, the length() function on Point3D is virtual, so compute it manually.
            const double length = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
            *data = FeatureValues<T>::convertToFeatureType(length);
        }
        return;
    }
    case BondFeature::ESTIMATED_BOND_LENGTH: {
        for (uint32_t i = 0; i < num_bonds; ++i, data += stride) {
            const uint32_t begin_atom = graph.bonds[i].beginAtomIdx;
            const uint32_t end_atom = graph.bonds[i].endAtomIdx;
            const int atomic_num1 = graph.atoms[begin_atom].atomicNum;
            const bool atom1_valid = (atomic_num1 >= 1 && atomic_num1 <= 118);
            const int atomic_num2 = graph.atoms[end_atom].atomicNum;
            const bool atom2_valid = (atomic_num2 >= 1 && atomic_num2 <= 118);
            assert(atom1_valid && atom2_valid);
            if (!atom1_valid || !atom2_valid) {
                *data = FeatureValues<T>::nan_value;
                continue;
            }

            const auto type = graph.bonds[i].bondType;
            if (type == RDKit::Bond::BondType::SINGLE) {
                // All atoms have a single bond length
                *data = FeatureValues<T>::convertToFeatureType(
                    single_bond_lengths[atomic_num1 - 1] + single_bond_lengths[atomic_num2 - 1]);
                continue;
            }
            if (type == RDKit::Bond::BondType::DOUBLE) {
                const double length1 = (double_bond_lengths[atomic_num1 - 1] >= 0) ?
                    double_bond_lengths[atomic_num1 - 1] : single_bond_lengths[atomic_num1 - 1];
                const double length2 = (double_bond_lengths[atomic_num2 - 1] >= 0) ?
                    double_bond_lengths[atomic_num2 - 1] : single_bond_lengths[atomic_num2 - 1];
                *data = FeatureValues<T>::convertToFeatureType(length1 + length2);
                continue;
            }
            if (type == RDKit::Bond::BondType::TRIPLE) {
                const double length1 = (triple_bond_lengths[atomic_num1 - 1] >= 0) ?
                    triple_bond_lengths[atomic_num1 - 1] : single_bond_lengths[atomic_num1 - 1];
                const double length2 = (triple_bond_lengths[atomic_num2 - 1] >= 0) ?
                    triple_bond_lengths[atomic_num2 - 1] : single_bond_lengths[atomic_num2 - 1];
                *data = FeatureValues<T>::convertToFeatureType(length1 + length2);
                continue;
            }
            if (type != RDKit::Bond::BondType::AROMATIC) {
                *data = FeatureValues<T>::nan_value;
            }

            // Aromatic case
            double length1 = single_bond_lengths[atomic_num1 - 1];
            double length2 = single_bond_lengths[atomic_num2 - 1];
            if (double_bond_lengths[atomic_num1] >= 0) {
                length1 = 0.5 * (length1 + double_bond_lengths[atomic_num1 - 1]);
            }
            if (double_bond_lengths[atomic_num2] >= 0) {
                length2 = 0.5 * (length2 + double_bond_lengths[atomic_num2 - 1]);
            }
            *data = FeatureValues<T>::convertToFeatureType(length1 + length2);
        }
        return;
    }
    default:
        // Missing implementation
        assert(0);
        for (uint32_t i = 0; i < num_bonds; ++i, data += stride) {
            *data = FeatureValues<T>::nan_value;
        }
        return;
    }
}

// Explicit instantiations, so that the function can be templated
// but still be used from other cpp files.
template void get_bond_float_feature<int16_t>(const GraphData& graph, int16_t* data, BondFeature feature, size_t stride);
template void get_bond_float_feature<float>(const GraphData& graph, float* data, BondFeature feature, size_t stride);
template void get_bond_float_feature<double>(const GraphData& graph, double* data, BondFeature feature, size_t stride);
