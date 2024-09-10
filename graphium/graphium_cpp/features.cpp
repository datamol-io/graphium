// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define DEBUG_LOGGING 0

#include "features.h"

#include "commute.h"
#include "electrostatic.h"
#include "float_features.h"
#include "graphormer.h"
#include "one_hot.h"
#include "random_walk.h"
#include "spectral.h"

#include <GraphMol/MolOps.h> // For RDKit's addHs
#include <GraphMol/DistGeomHelpers/Embedder.h> // For RDKit's EmbedMolecule

#include <unordered_map>

static GraphData read_graph(const std::string& smiles_string, bool explicit_H) {
    std::unique_ptr<RDKit::RWMol> mol{ parse_mol(smiles_string, explicit_H) };

    if (!mol) {
        return GraphData{ 0, std::unique_ptr<CompactAtom[]>(), 0, std::unique_ptr<CompactBond[]>(), std::move(mol) };
    }

    const size_t num_atoms = mol->getNumAtoms();
    const size_t num_bonds = mol->getNumBonds();
#if DEBUG_LOGGING
    printf("# atoms = %zu\n# bonds = %zu\n", num_atoms, num_bonds);
#endif
#if REPORT_STATS
    ++statsMolAtomCounts[(num_atoms >= STATS_NUM_MOL_ATOM_COUNTS) ? (STATS_NUM_MOL_ATOM_COUNTS - 1) : num_atoms];
    ++statsMolBondCounts[(num_bonds >= STATS_NUM_MOL_BOND_COUNTS) ? (STATS_NUM_MOL_BOND_COUNTS - 1) : num_bonds];
    statsTotalNumAtoms += num_atoms;
    statsTotalNumBonds += num_bonds;
#endif

#if ORDER_ATOMS
    // Determine a canonical ordering of the atoms, if desired.
    std::vector<unsigned int> atomOrder;
    atomOrder.reserve(num_atoms);
    RDKit::Canon::rankMolAtoms(*mol, atomOrder);
    assert(atomOrder.size() == num_atoms);
#endif

    // Allocate an array of atom data, and fill it from the RDKit atom data.
    std::unique_ptr<CompactAtom[]> atoms(new CompactAtom[num_atoms]);
    for (size_t atomIdx = 0; atomIdx < num_atoms; ++atomIdx) {
        const RDKit::Atom* const atom = mol->getAtomWithIdx(atomIdx);
        auto atomicNum = atom->getAtomicNum();
        auto totalDegree = atom->getTotalDegree();
        auto formalCharge = atom->getFormalCharge();
        const RDKit::Atom::ChiralType chiralType = atom->getChiralTag();
        auto totalNumHs = atom->getTotalNumHs();
        const RDKit::Atom::HybridizationType hybridization = atom->getHybridization();

        const bool isAromatic = atom->getIsAromatic();
#if REPORT_STATS
        ++statsElementCounts[(atomicNum < 0 || atomicNum >= STATS_NUM_ELEMENTS) ? (STATS_NUM_ELEMENTS - 1) : atomicNum];
        ++statsDegreeCounts[(totalDegree < 0 || totalDegree >= STATS_NUM_DEGREES) ? (STATS_NUM_DEGREES - 1) : totalDegree];
        size_t formalChargeIndex = formalCharge + int(STATS_CHARGE_OFFSET);
        if (formalCharge < -int(STATS_CHARGE_OFFSET)) {
            formalChargeIndex = 0;
        }
        else if (formalCharge > int(STATS_CHARGE_OFFSET)) {
            formalChargeIndex = STATS_NUM_CHARGES - 1;
        }

        ++statsChargeCounts[formalChargeIndex];
        ++statsChiralityCounts[(size_t(chiralType) >= STATS_NUM_CHIRALITIES) ? (STATS_NUM_CHIRALITIES - 1) : size_t(chiralType)];
        ++statsHCounts[(totalNumHs < 0 || totalNumHs >= STATS_NUM_HS) ? (STATS_NUM_HS - 1) : totalNumHs];
        ++statsHybridizationCounts[(size_t(hybridization) >= STATS_NUM_HYBRIDIZATIONS) ? (STATS_NUM_HYBRIDIZATIONS - 1) : size_t(hybridization)];
        statsAromaticAtomCount += (isAromatic ? 1 : 0);
#endif
        const double mass = atom->getMass();

#if ORDER_ATOMS
        const size_t destAtomIdx = atomOrder[atomIdx];
#else
        const size_t destAtomIdx = atomIdx;
#endif
        atoms[destAtomIdx] = CompactAtom{
            uint8_t(atomicNum),
            uint8_t(totalDegree),
            int8_t(formalCharge),
            uint8_t(chiralType),
            uint8_t(totalNumHs),
            uint8_t(hybridization),
            isAromatic,
            float(mass)
        };
#if DEBUG_LOGGING
        printf(
            "atom[%zu] = {%zu, %u, %d, %u, %u, %u, %s, %f}\n",
            destAtomIdx,
            int(atomicNum),
            int(totalDegree),
            int(formalCharge),
            int(chiralType),
            int(totalNumHs),
            int(hybridization),
            isAromatic ? "true" : "false",
            mass
        );
#endif
    }

    // Allocate an array of bond data, and fill it from the RDKit bond data.
    std::unique_ptr<CompactBond[]> bonds(new CompactBond[num_bonds]);
    const RDKit::RingInfo* const ringInfo = mol->getRingInfo();
    for (size_t bondIdx = 0; bondIdx < num_bonds; ++bondIdx) {
        const RDKit::Bond* const bond = mol->getBondWithIdx(bondIdx);
        const RDKit::Bond::BondType bondType = bond->getBondType();
        const bool isConjugated = bond->getIsConjugated();
        // TODO: Verify that it's the same index as bond->getIdx()
        const bool isInRing = (ringInfo->numBondRings(bondIdx) != 0);
        const RDKit::Bond::BondStereo stereo = bond->getStereo();

#if REPORT_STATS
        ++statsBondTypeCounts[(size_t(bondType) >= STATS_NUM_BOND_TYPES) ? (STATS_NUM_BOND_TYPES - 1) : size_t(bondType)];
        ++statsBondStereoCounts[(size_t(stereo) >= STATS_NUM_BOND_STEREOS) ? (STATS_NUM_BOND_STEREOS - 1) : size_t(stereo)];
        statsConjugatedBondCount += (isConjugated ? 1 : 0);
        statsBondInRingCount += (isInRing ? 1 : 0);
#endif

        auto beginAtomIdx = bond->getBeginAtomIdx();
        auto endAtomIdx = bond->getEndAtomIdx();
#if ORDER_ATOMS
        beginAtomIdx = atomOrder[beginAtomIdx];
        endAtomIdx = atomOrder[endAtomIdx];
#endif
        bonds[bondIdx] = CompactBond{
            uint8_t(bondType),
            isConjugated,
            isInRing,
            uint8_t(stereo),
            beginAtomIdx,
            endAtomIdx
        };
#if DEBUG_LOGGING
        printf(
            "bond[%zu] = {%u, %s, %s, %u, {%u, %u}}\n",
            bondIdx,
            int(bondType),
            isConjugated ? "true" : "false",
            isInRing ? "true" : "false",
            int(stereo),
            beginAtomIdx,
            endAtomIdx
        );
#endif
    }

    // Return a GraphData structure, taking ownership of the atom and bond data arrays.
    return GraphData{ num_atoms, std::move(atoms), num_bonds, std::move(bonds), std::move(mol) };
}

// This is a structure for managing the adjacency data (CSR format) for use by randomSubgraph.
struct NeighbourData {
    // This owns the data of all 3 arrays, which are actually a single, contiguous allocation.
    std::unique_ptr<uint32_t[]> deleter;

    // This is an array of indices into the other two arrays, indicating where
    // each atom's neighbours start, including the first entry being 0 for the start of
    // atom 0, and the num_atoms entry being 2*num_bonds (2x because each bond is on 2 atoms),
    // so there are num_atoms+1 entries.  The number of neighbours of an atom i is
    // neighbour_starts[i+1]-neighbour_starts[i]
    const uint32_t* neighbour_starts;

    // The neighbour atom for each bond, with each atom having an entry for each of
    // its neighbours, so each bond occurs twice.
    const uint32_t* neighbours;

    // This is in the same order as neighbours, but indicates the index of the bond.
    // Each bond occurs twice, so each number occurs twice.
    const uint32_t* bond_indices;
};

// Construct a NeighbourData structure representing the molecule's graph in CSR format.
static NeighbourData construct_neighbours(const GraphData& graph) {
    const uint32_t num_atoms = graph.num_atoms;
    const uint32_t num_bonds = graph.num_bonds;
    // Do a single allocation for all 3 arrays.
    std::unique_ptr<uint32_t[]> deleter(new uint32_t[num_atoms + 1 + 4 * num_bonds]);

    uint32_t* neighbour_starts = deleter.get();
    for (uint32_t i = 0; i <= num_atoms; ++i) {
        neighbour_starts[i] = 0;
    }

    // First, get atom neighbour counts
    const CompactBond* const bonds = graph.bonds.get();
    for (uint32_t i = 0; i < num_bonds; ++i) {
        uint32_t a = bonds[i].beginAtomIdx;
        uint32_t b = bonds[i].endAtomIdx;
        // NOTE: +1 is because first entry will stay zero.
        ++neighbour_starts[a + 1];
        ++neighbour_starts[b + 1];
    }

    // Find the starts by partial-summing the neighbour counts.
    // NOTE: +1 is because first entry will stay zero.
    std::partial_sum(neighbour_starts + 1, neighbour_starts + 1 + num_atoms, neighbour_starts + 1);

    // Fill in the neighbours and bond_indices arrays.
    uint32_t* neighbours = neighbour_starts + num_atoms + 1;
    uint32_t* bond_indices = neighbours + 2 * num_bonds;
    for (uint32_t i = 0; i < num_bonds; ++i) {
        uint32_t a = bonds[i].beginAtomIdx;
        uint32_t b = bonds[i].endAtomIdx;

        uint32_t ai = neighbour_starts[a];
        neighbours[ai] = b;
        bond_indices[ai] = i;
        ++neighbour_starts[a];

        uint32_t bi = neighbour_starts[b];
        neighbours[bi] = a;
        bond_indices[bi] = i;
        ++neighbour_starts[b];
    }

    // Shift neighbour_starts forward one after incrementing it.
    uint32_t previous = 0;
    for (uint32_t i = 0; i < num_atoms; ++i) {
        uint32_t next = neighbour_starts[i];
        neighbour_starts[i] = previous;
        previous = next;
    }

    // NeighbourData takes ownership of the memory.
    return NeighbourData{ std::move(deleter), neighbour_starts, neighbours, bond_indices };
}

// This fills in 3 values for each atom
template<typename T>
at::Tensor get_conformer_features(
    RDKit::ROMol &mol,
    bool already_has_Hs,
    c10::ScalarType dtype,
    MaskNaNStyle mask_nan_style,
    T mask_nan_value,
    int64_t &num_nans,
    const std::string& smiles_string) {

    const size_t n = mol.getNumAtoms();
    std::unique_ptr<T[]> conformer_data(new T[3 * n]);
    T* data = conformer_data.get();

    std::unique_ptr<RDKit::RWMol> mol_with_Hs_added;
    RDKit::ROMol* mol_with_Hs = &mol;
    if (mol.beginConformers() == mol.endConformers()) {
        // No conformers.
        // Before generating conformers, it's recommended to add Hs explicitly.
        if (!already_has_Hs) {
            // Add Hs.  They're added at the end, so the original atoms
            // will have the same indices as before.
            mol_with_Hs_added.reset(new RDKit::RWMol(mol));
            RDKit::MolOps::addHs(*mol_with_Hs_added);
            mol_with_Hs = mol_with_Hs_added.get();
        }
        
        // Default Python arguments to EmbedMolecule
        int conformer_id = RDKit::DGeomHelpers::EmbedMolecule(
            *mol_with_Hs,
            0,      // maxIterations
            -1,     // seed
            true,   // clearConfs
            false,  // useRandomCoords
            2.0,    // boxSizeMult
            true,   // randNedEig
            1,      // numZeroFail
            nullptr,// coordMap
            1e-3,   // optimizerForceTol
            false,  // ignoreSmoothingFailures
            true,   // enforceChirality
            true,   // useExpTorsionAnglePrefs (default in Python; non-default in C++)
            true,   // useBasicKnowledge (default in Python; non-default in C++)
            false,  // verbose
            5.0,    // basinThresh
            false,  // onlyHeavyAtomsForRMS
            1,      // ETversion
            false,  // useSmallRingTorsions
            false,  // useMacrocycleTorsions
            false   // useMacrocycle14config
        );
        
        if (conformer_id == -1) {
            // Custom arguments as fallback
            RDKit::DGeomHelpers::EmbedMolecule(
                *mol_with_Hs,
                0,      // maxIterations
                -1,     // seed
                true,   // clearConfs
                false,  // useRandomCoords (TODO: consider using true)
                2.0,    // boxSizeMult
                true,   // randNedEig
                1,      // numZeroFail
                nullptr,// coordMap
                0.1,    // optimizerForceTol (changed)
                true,   // ignoreSmoothingFailures (changed)
                false,  // enforceChirality (changed)
                true,   // useExpTorsionAnglePrefs (default in Python; non-default in C++)
                true,   // useBasicKnowledge (default in Python; non-default in C++)
                false,  // verbose
                5.0,    // basinThresh
                false,  // onlyHeavyAtomsForRMS
                1,      // ETversion
                false,  // useSmallRingTorsions
                false,  // useMacrocycleTorsions
                false   // useMacrocycle14config
            );
        }
    }
    if (mol_with_Hs->beginConformers() == mol_with_Hs->endConformers()) {
        // Still no conformers: treat as NaN
        for (size_t i = 0; i < 3 * n; ++i) {
            data[i] = mask_nan_value;
        }
        if (mask_nan_style == MaskNaNStyle::REPORT) {
            num_nans += 3*n;
        }
        printf("Warning: Couldn't compute conformer for molecule \"%s\"\n", smiles_string.c_str());
    }
    else {
        const RDKit::Conformer& conformer = mol_with_Hs->getConformer();
        const auto& positions = conformer.getPositions();
        assert(positions.size() >= n);
        for (size_t i = 0; i < n; ++i, data += 3) {
            const auto& position = positions[i];
            data[0] = FeatureValues<T>::convertToFeatureType(position.x);
            data[1] = FeatureValues<T>::convertToFeatureType(position.y);
            data[2] = FeatureValues<T>::convertToFeatureType(position.z);
        }

        num_nans += mask_nans(data, 3 * n, mask_nan_style, mask_nan_value);
    }

    const int64_t dims[1] = { int64_t(3 * n) };
    return torch_tensor_from_array<T>(std::move(conformer_data), dims, 1, dtype);
}

static const std::unordered_map<std::string, int64_t> atom_float_name_to_enum{
    {std::string("atomic-number"), int64_t(AtomFloatFeature::ATOMIC_NUMBER)},
    {std::string("mass"), int64_t(AtomFloatFeature::MASS)},
    {std::string("weight"), int64_t(AtomFloatFeature::MASS)},
    {std::string("valence"), int64_t(AtomFloatFeature::VALENCE)},
    {std::string("total-valence"), int64_t(AtomFloatFeature::VALENCE)},
    {std::string("implicit-valence"), int64_t(AtomFloatFeature::IMPLICIT_VALENCE)},
    {std::string("hybridization"), int64_t(AtomFloatFeature::HYBRIDIZATION)},
    {std::string("chirality"), int64_t(AtomFloatFeature::CHIRALITY)},
    {std::string("aromatic"), int64_t(AtomFloatFeature::AROMATIC)},
    {std::string("ring"), int64_t(AtomFloatFeature::IN_RING)},
    {std::string("in-ring"), int64_t(AtomFloatFeature::IN_RING)},
    {std::string("min-ring"), int64_t(AtomFloatFeature::MIN_RING)},
    {std::string("max-ring"), int64_t(AtomFloatFeature::MAX_RING)},
    {std::string("num-ring"), int64_t(AtomFloatFeature::NUM_RING)},
    {std::string("degree"), int64_t(AtomFloatFeature::DEGREE)},
    {std::string("radical-electron"), int64_t(AtomFloatFeature::RADICAL_ELECTRON)},
    {std::string("formal-charge"), int64_t(AtomFloatFeature::FORMAL_CHARGE)},
    {std::string("vdw-radius"), int64_t(AtomFloatFeature::VDW_RADIUS)},
    {std::string("covalent-radius"), int64_t(AtomFloatFeature::COVALENT_RADIUS)},
    {std::string("electronegativity"), int64_t(AtomFloatFeature::ELECTRONEGATIVITY)},
    {std::string("ionization"), int64_t(AtomFloatFeature::IONIZATION)},
    {std::string("first-ionization"), int64_t(AtomFloatFeature::IONIZATION)},
    {std::string("melting-point"), int64_t(AtomFloatFeature::MELTING_POINT)},
    {std::string("metal"), int64_t(AtomFloatFeature::METAL)},
    {std::string("group"), int64_t(AtomFloatFeature::GROUP)},
    {std::string("period"), int64_t(AtomFloatFeature::PERIOD)},
    {std::string("single-bond"), int64_t(AtomFloatFeature::SINGLE_BOND)},
    {std::string("aromatic-bond"), int64_t(AtomFloatFeature::AROMATIC_BOND)},
    {std::string("double-bond"), int64_t(AtomFloatFeature::DOUBLE_BOND)},
    {std::string("triple-bond"), int64_t(AtomFloatFeature::TRIPLE_BOND)},
    {std::string("is-carbon"), int64_t(AtomFloatFeature::IS_CARBON)},
};

at::Tensor atom_float_feature_names_to_tensor(const std::vector<std::string>& features) {
    const size_t num_features = features.size();
    std::unique_ptr<int64_t[]> feature_enum_values(new int64_t[num_features]);
    for (size_t i = 0; i < num_features; ++i) {
        auto it = atom_float_name_to_enum.find(features[i]);
        if (it != atom_float_name_to_enum.end()) {
            feature_enum_values[i] = it->second;
        }
        else {
            feature_enum_values[i] = int64_t(AtomFloatFeature::UNKNOWN);
        }
    }
    const int64_t dims[1] = { int64_t(num_features) };
    return torch_tensor_from_array<int64_t>(std::move(feature_enum_values), dims, 1, c10::ScalarType::Long);
}

static const std::unordered_map<std::string, int64_t> atom_onehot_name_to_enum{
    {std::string("atomic-number"), int64_t(AtomOneHotFeature::ATOMIC_NUM)},
    {std::string("degree"), int64_t(AtomOneHotFeature::DEGREE)},
    {std::string("valence"), int64_t(AtomOneHotFeature::VALENCE)},
    {std::string("total-valence"), int64_t(AtomOneHotFeature::VALENCE)},
    {std::string("implicit-valence"), int64_t(AtomOneHotFeature::IMPLICIT_VALENCE)},
    {std::string("hybridization"), int64_t(AtomOneHotFeature::HYBRIDIZATION)},
    {std::string("chirality"), int64_t(AtomOneHotFeature::CHIRALITY)},
    {std::string("phase"), int64_t(AtomOneHotFeature::PHASE)},
    {std::string("type"), int64_t(AtomOneHotFeature::TYPE)},
    {std::string("group"), int64_t(AtomOneHotFeature::GROUP)},
    {std::string("period"), int64_t(AtomOneHotFeature::PERIOD)},
};

at::Tensor atom_onehot_feature_names_to_tensor(const std::vector<std::string>& features) {
    const size_t num_features = features.size();
    std::unique_ptr<int64_t[]> feature_enum_values(new int64_t[num_features]);
    for (size_t i = 0; i < num_features; ++i) {
        auto it = atom_onehot_name_to_enum.find(features[i]);
        if (it != atom_onehot_name_to_enum.end()) {
            feature_enum_values[i] = it->second;
        }
        else {
            feature_enum_values[i] = int64_t(AtomOneHotFeature::UNKNOWN);
        }
    }
    const int64_t dims[1] = { int64_t(num_features) };
    return torch_tensor_from_array<int64_t>(std::move(feature_enum_values), dims, 1, c10::ScalarType::Long);
}

static const std::unordered_map<std::string, int64_t> bond_name_to_enum{
    {std::string("bond-type-onehot"), int64_t(BondFeature::TYPE_ONE_HOT)},
    {std::string("bond-type-float"), int64_t(BondFeature::TYPE_FLOAT)},
    {std::string("stereo"), int64_t(BondFeature::STEREO_ONE_HOT)},
    {std::string("in-ring"), int64_t(BondFeature::IN_RING)},
    {std::string("conjugated"), int64_t(BondFeature::CONJUGATED)},
    {std::string("conformer-bond-length"), int64_t(BondFeature::CONFORMER_BOND_LENGTH)},
    {std::string("estimated-bond-length"), int64_t(BondFeature::ESTIMATED_BOND_LENGTH)},
};

at::Tensor bond_feature_names_to_tensor(const std::vector<std::string>& features) {
    const size_t num_features = features.size();
    std::unique_ptr<int64_t[]> feature_enum_values(new int64_t[num_features]);
    for (size_t i = 0; i < num_features; ++i) {
        auto it = bond_name_to_enum.find(features[i]);
        if (it != bond_name_to_enum.end()) {
            feature_enum_values[i] = it->second;
        }
        else {
            feature_enum_values[i] = int64_t(BondFeature::UNKNOWN);
        }
    }
    const int64_t dims[1] = { int64_t(num_features) };
    return torch_tensor_from_array<int64_t>(std::move(feature_enum_values), dims, 1, c10::ScalarType::Long);
}

static const std::unordered_map<std::string, int64_t> positional_name_to_enum{
    {std::string("laplacian_eigvec"), int64_t(PositionalFeature::LAPLACIAN_EIGENVEC)},
    {std::string("laplacian_eigval"), int64_t(PositionalFeature::LAPLACIAN_EIGENVAL)},
    {std::string("rw_return_probs"), int64_t(PositionalFeature::RW_RETURN_PROBS)},
    {std::string("rw_transition_probs"), int64_t(PositionalFeature::RW_TRANSITION_PROBS)},
    {std::string("electrostatic"), int64_t(PositionalFeature::ELECTROSTATIC)},
    {std::string("commute"), int64_t(PositionalFeature::COMMUTE)},
    {std::string("graphormer"), int64_t(PositionalFeature::GRAPHORMER)},
};

static const std::unordered_map<std::string, int64_t> feature_level_to_enum{
    {std::string("node"), int64_t(FeatureLevel::NODE)},
    {std::string("edge"), int64_t(FeatureLevel::EDGE)},
    {std::string("nodepair"), int64_t(FeatureLevel::NODEPAIR)},
    {std::string("graph"), int64_t(FeatureLevel::GRAPH)},
};

static const std::unordered_map<std::string, int64_t> normalization_to_enum{
    {std::string("none"), int64_t(Normalization::NONE)},
    {std::string("inv"), int64_t(Normalization::INVERSE)},
    {std::string("sym"), int64_t(Normalization::SYMMETRIC)},
};

std::pair<std::vector<std::string>,at::Tensor> positional_feature_options_to_tensor(
    const pybind11::dict& dict) {
    size_t num_features = 0;
    size_t num_values = 0;
    for (const auto& pair : dict) {
        // The string keys (pair.first) of the outer dictionary aren't needed for this
        if (!pybind11::isinstance<pybind11::dict>(pair.second)) {
            continue;
        }
        pybind11::dict feature_dict = pair.second.cast<pybind11::dict>();
        pybind11::handle feature_name_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "pos_type"));
        pybind11::handle feature_level_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "pos_level"));
        if (!feature_name_handle || !feature_level_handle) {
            continue;
        }
        std::string feature_name{ pybind11::str(feature_name_handle) };
        std::string feature_level{ pybind11::str(feature_level_handle) };

        auto feature_it = positional_name_to_enum.find(feature_name);
        auto level_it = feature_level_to_enum.find(feature_level);
        if (feature_it == positional_name_to_enum.end() || level_it == feature_level_to_enum.end()) {
            continue;
        }

        PositionalFeature feature = PositionalFeature(feature_it->second);
        switch (feature) {
        case PositionalFeature::LAPLACIAN_EIGENVEC:
        case PositionalFeature::LAPLACIAN_EIGENVAL: {
            // Required int num_pos
            pybind11::handle num_pos_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "num_pos"));
            if (!num_pos_handle || !pybind11::isinstance<pybind11::int_>(num_pos_handle)) {
                break;
            }
            // Optional string normalization
            pybind11::handle normalization_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "normalization"));
            if (normalization_handle) {
                if (!pybind11::isinstance<pybind11::str>(normalization_handle)) {
                    break;
                }
                std::string normalization_name{ pybind11::str(normalization_handle) };
                if (!normalization_to_enum.contains(normalization_name)) {
                    break;
                }
            }
            // Optional bool disconnected_comp
            pybind11::handle disconnected_comp_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "disconnected_comp"));
            if (disconnected_comp_handle && !pybind11::isinstance<pybind11::bool_>(disconnected_comp_handle)) {
                break;
            }
            num_values += 3 + 3;
            ++num_features;
            break;
        }
        case PositionalFeature::RW_RETURN_PROBS:
        case PositionalFeature::RW_TRANSITION_PROBS: {
            pybind11::handle ksteps_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "ksteps"));
            if (!ksteps_handle) {
                break;
            }
            int64_t power_count = 0;
            if (pybind11::isinstance<pybind11::int_>(ksteps_handle)) {
                power_count = int64_t(ksteps_handle.cast<pybind11::int_>());
            }
            else if (pybind11::isinstance<pybind11::list>(ksteps_handle)) {
                power_count = ksteps_handle.cast<pybind11::list>().size();
            }
            if (power_count < 1) {
                break;
            }
            pybind11::handle space_dim_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "space_dim"));
            if (space_dim_handle && !pybind11::isinstance<pybind11::int_>(space_dim_handle)) {
                break;
            }
            num_values += 3 + 1 + power_count;
            ++num_features;
            break;
        }
        case PositionalFeature::ELECTROSTATIC:
        case PositionalFeature::COMMUTE:
        case PositionalFeature::GRAPHORMER:
            num_values += 3;
            ++num_features;
            break;
        }
    }

    std::unique_ptr<int64_t[]> values(new int64_t[num_values]);
    std::vector<std::string> names(num_features);

    size_t prev_feature_index = 0;
    size_t feature_index = 0;
    size_t value_index = 0;
    for (const auto& pair : dict) {
        // The string keys (pair.first) of the outer dictionary aren't needed for this
        if (!pybind11::isinstance<pybind11::dict>(pair.second)) {
            continue;
        }
        pybind11::dict feature_dict = pair.second.cast<pybind11::dict>();
        pybind11::handle feature_name_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "pos_type"));
        pybind11::handle feature_level_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "pos_level"));
        if (!feature_name_handle || !feature_level_handle) {
            continue;
        }
        std::string feature_name{ pybind11::str(feature_name_handle) };
        std::string feature_level{ pybind11::str(feature_level_handle) };

        auto feature_it = positional_name_to_enum.find(feature_name);
        auto level_it = feature_level_to_enum.find(feature_level);
        if (feature_it == positional_name_to_enum.end() || level_it == feature_level_to_enum.end()) {
            continue;
        }

        PositionalFeature feature = PositionalFeature(feature_it->second);
        switch (feature) {
        case PositionalFeature::LAPLACIAN_EIGENVEC:
        case PositionalFeature::LAPLACIAN_EIGENVAL: {
            // Required int num_pos
            pybind11::handle num_pos_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "num_pos"));
            if (!num_pos_handle || !pybind11::isinstance<pybind11::int_>(num_pos_handle)) {
                continue;
            }
            // Optional string normalization
            pybind11::handle normalization_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "normalization"));
            Normalization normalization = Normalization::NONE;
            if (normalization_handle) {
                if (!pybind11::isinstance<pybind11::str>(normalization_handle)) {
                    continue;
                }
                std::string normalization_name{ pybind11::str(normalization_handle) };
                auto it = normalization_to_enum.find(normalization_name);
                if (it == normalization_to_enum.end()) {
                    continue;
                }
                normalization = Normalization(it->second);
            }
            // Optional bool disconnected_comp
            pybind11::handle disconnected_comp_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "disconnected_comp"));
            if (disconnected_comp_handle && !pybind11::isinstance<pybind11::bool_>(disconnected_comp_handle)) {
                continue;
            }
            values[value_index] = feature_it->second;
            values[value_index + 1] = 3;
            values[value_index + 2] = level_it->second;
            values[value_index + 3] = int64_t(num_pos_handle.cast<pybind11::int_>());
            values[value_index + 4] = int64_t(normalization);
            values[value_index + 5] = disconnected_comp_handle ? bool(disconnected_comp_handle.cast<pybind11::bool_>()) : true;
            value_index += 3 + 3;
            ++feature_index;
            break;
        }
        case PositionalFeature::RW_RETURN_PROBS:
        case PositionalFeature::RW_TRANSITION_PROBS: {
            // Required int or list[int] ksteps
            pybind11::handle ksteps_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "ksteps"));
            if (!ksteps_handle) {
                continue;
            }
            int64_t power_count = 0;
            if (pybind11::isinstance<pybind11::int_>(ksteps_handle)) {
                // Integer means use all powers from 1 up to this value, inclusive.
                power_count = int64_t(ksteps_handle.cast<pybind11::int_>());
            }
            else if (pybind11::isinstance<pybind11::list>(ksteps_handle)) {
                power_count = ksteps_handle.cast<pybind11::list>().size();
            }
            if (power_count < 1) {
                break;
            }
            // Optional int space_dim
            pybind11::handle space_dim_handle = pybind11::handle(PyDict_GetItemString(feature_dict.ptr(), "space_dim"));
            if (space_dim_handle && !pybind11::isinstance<pybind11::int_>(space_dim_handle)) {
                break;
            }
            values[value_index] = feature_it->second;
            values[value_index + 1] = 1 + power_count;
            values[value_index + 2] = level_it->second;

            int64_t space_dim = space_dim_handle ? int64_t(space_dim_handle.cast<pybind11::int_>()) : 0;
            values[value_index + 3] = space_dim;
            if (pybind11::isinstance<pybind11::int_>(ksteps_handle)) {
                for (int64_t power = 1; power <= power_count; ++power) {
                    values[value_index + 3 + power] = power;
                }
            }
            else if (pybind11::isinstance<pybind11::list>(ksteps_handle)) {
                size_t power_index = 0;
                int64_t prev_power = 0;
                for(const auto item : ksteps_handle.cast<pybind11::list>()) {
                    int64_t next_power = pybind11::isinstance<pybind11::int_>(item) ? int64_t(item.cast<pybind11::int_>()) : prev_power;
                    if (next_power < prev_power) {
                        // Force the integers to be ascending
                        next_power = prev_power;
                    }
                    values[value_index + 3 + 1 + power_index] = next_power;
                    prev_power = next_power;
                    ++power_index;
                }
            }
            value_index += 3 + 1 + power_count;
            ++feature_index;
            break;
        }
        case PositionalFeature::ELECTROSTATIC:
        case PositionalFeature::COMMUTE:
        case PositionalFeature::GRAPHORMER:
            values[value_index] = feature_it->second;
            values[value_index + 1] = 0;
            values[value_index + 2] = level_it->second;
            value_index += 3;
            ++feature_index;
            break;
        }
        if (feature_index != prev_feature_index) {
            names[prev_feature_index] = (level_it->second == int64_t(FeatureLevel::NODE)) ? feature_name : (feature_level + std::string("_") + feature_name);
            ++prev_feature_index;
        }
    }
    assert(feature_index == num_features && prev_feature_index == num_features && value_index == num_values);

    const int64_t dims[1] = { int64_t(num_values) };
    return std::make_pair(
        std::move(names),
        torch_tensor_from_array<int64_t>(std::move(values), dims, 1, c10::ScalarType::Long));
}

template<typename T>
at::Tensor create_edge_weights(
    const GraphData& graph,
    bool duplicate_edges,
    bool add_self_loop,
    bool use_bonds_weights,
    c10::ScalarType dtype) {

    const size_t edge_coo_count = (duplicate_edges ? 2*graph.num_bonds : graph.num_bonds) +
                                    (add_self_loop ? graph.num_atoms : 0);
    std::unique_ptr<T[]> edge_weights(new T[edge_coo_count]);

    // TODO: Use use_bonds_weights to optionally give weights
    // in same order as other edge features
    for (size_t i = 0; i < edge_coo_count; ++i) {
        edge_weights[i] = FeatureValues<T>::one;
    }

    const int64_t dims[1] = { int64_t(edge_coo_count) };
    return torch_tensor_from_array<T>(std::move(edge_weights), dims, 1, dtype);
}

template<typename T>
at::Tensor create_atom_features(
    const GraphData& graph,
    const at::Tensor& atom_property_list_onehot,
    const at::Tensor& atom_property_list_float,
    bool offset_carbon,
    c10::ScalarType dtype,
    MaskNaNStyle mask_nan_style,
    T mask_nan_value,
    int64_t &num_nans) {

    const size_t num_onehot_properties = (atom_property_list_onehot.scalar_type() == c10::ScalarType::Long && atom_property_list_onehot.ndimension() == 1) ? atom_property_list_onehot.size(0) : 0;
    // NOTE: If TensorBase::data_ptr is ever removed, change it to TensorBase::const_data_ptr.
    // Some torch version being used doesn't have const_data_ptr yet.
    const int64_t* const property_list_onehot = (num_onehot_properties != 0) ? atom_property_list_onehot.data_ptr<int64_t>() : nullptr;
    const size_t num_float_properties = (atom_property_list_float.scalar_type() == c10::ScalarType::Long && atom_property_list_float.ndimension() == 1) ? atom_property_list_float.size(0) : 0;
    const int64_t* const property_list_float = (num_float_properties != 0) ? atom_property_list_float.data_ptr<int64_t>() : nullptr;

    size_t single_atom_float_count = num_float_properties;
    for (size_t i = 0; i < num_onehot_properties; ++i) {
        const int64_t property = property_list_onehot[i];
        single_atom_float_count += get_one_hot_atom_feature_size(AtomOneHotFeature(property));
    }
    const size_t atom_float_count = single_atom_float_count * graph.num_atoms;

    std::unique_ptr<T[]> atom_data(new T[atom_float_count]);

    T* current_atom_data = atom_data.get();

    for (size_t i = 0; i < num_float_properties; ++i) {
        const int64_t property = property_list_float[i];
        get_atom_float_feature(graph, current_atom_data, AtomFloatFeature(property), single_atom_float_count, offset_carbon);
        ++current_atom_data;
    }
    for (size_t i = 0; i < num_onehot_properties; ++i) {
        const int64_t property = property_list_onehot[i];
        current_atom_data += get_one_hot_atom_feature(graph, current_atom_data, AtomOneHotFeature(property), single_atom_float_count);
    }

    num_nans += mask_nans(atom_data.get(), atom_float_count, mask_nan_style, mask_nan_value);

    const int64_t dims[2] = { int64_t(graph.num_atoms), int64_t(single_atom_float_count) };
    return torch_tensor_from_array<T>(std::move(atom_data), dims, 2, dtype);
}

template<typename T>
at::Tensor create_bond_features(
    const GraphData& graph,
    const at::Tensor& bond_property_list,
    const bool duplicate_edges,
    bool add_self_loop,
    c10::ScalarType dtype,
    MaskNaNStyle mask_nan_style,
    T mask_nan_value,
    int64_t& num_nans) {

    const size_t num_properties = (bond_property_list.scalar_type() == c10::ScalarType::Long && bond_property_list.ndimension() == 1) ? bond_property_list.size(0) : 0;
    const int64_t* const property_list = (num_properties != 0) ? bond_property_list.data_ptr<int64_t>() : nullptr;

    size_t single_bond_float_count = 0;
    for (size_t i = 0; i < num_properties; ++i) {
        const int64_t property = property_list[i];
        if (BondFeature(property) == BondFeature::TYPE_ONE_HOT || BondFeature(property) == BondFeature::STEREO_ONE_HOT) {
            single_bond_float_count += get_one_hot_bond_feature_size(BondFeature(property));
        }
        else {
            ++single_bond_float_count;
        }
    }
    // add_self_loop is only supported if duplicating edges
    add_self_loop = add_self_loop && duplicate_edges;
    size_t total_edge_count = graph.num_bonds;
    if (duplicate_edges) {
        total_edge_count = 2*total_edge_count + size_t(add_self_loop);
    }
    const size_t bond_float_count = single_bond_float_count * total_edge_count;

    std::unique_ptr<T[]> bond_data(new T[bond_float_count]);

    T* current_bond_data = bond_data.get();

    // This is the stride length (in floats) for each unique bond
    const size_t duplicated_bond_float_count = duplicate_edges ? (2*single_bond_float_count) : single_bond_float_count;

    for (size_t i = 0; i < num_properties; ++i) {
        const int64_t property = property_list[i];
        if (BondFeature(property) == BondFeature::TYPE_ONE_HOT || BondFeature(property) == BondFeature::STEREO_ONE_HOT) {
            current_bond_data += get_one_hot_bond_feature(graph, current_bond_data, BondFeature(property), duplicated_bond_float_count);
        }
        else {
            get_bond_float_feature(graph, current_bond_data, BondFeature(property), duplicated_bond_float_count);
            ++current_bond_data;
        }
    }

    if (duplicate_edges) {
        current_bond_data = bond_data.get();
        // Duplicate the data for each bond
        for (size_t i = 0; i < graph.num_bonds; ++i) {
            for (size_t j = 0; j < single_bond_float_count; ++j) {
                current_bond_data[j+single_bond_float_count] = current_bond_data[j];
            }
            current_bond_data += duplicated_bond_float_count;
        }
        if (add_self_loop) {
            // Self loops don't have valid bond data, but don't treat them as NaNs.
            // Fill with zeros, instead.
            memset(current_bond_data, 0, graph.num_atoms * graph.num_atoms);
        }
    }

    num_nans += mask_nans(bond_data.get(), bond_float_count, mask_nan_style, mask_nan_value);

    int64_t dims[2] = { int64_t(total_edge_count), int64_t(single_bond_float_count) };
    return torch_tensor_from_array<T>(std::move(bond_data), dims, 2, dtype);
}

template<typename OUT_T, typename IN_T>
void node_to_edge(
    std::unique_ptr<OUT_T[]>& output_ptr,
    size_t& floats_per_half_edge,
    const IN_T* input,
    const size_t n,
    const size_t floats_per_node,
    const GraphData& graph) {

    // Edge order must be consistent with the edges in the graph,
    // which is not necessarily lexicographic order.
    const size_t num_half_edges = 2*graph.num_bonds;
    floats_per_half_edge = 2 * floats_per_node;
    output_ptr.reset(new OUT_T[num_half_edges * 2 * floats_per_node]);
    OUT_T* output = output_ptr.get();
    for (size_t bond = 0; bond < graph.num_bonds; ++bond, output += 2*floats_per_half_edge) {
        const size_t atomi = graph.bonds[bond].beginAtomIdx;
        const size_t atomj = graph.bonds[bond].endAtomIdx;
        const IN_T* input_i = input + atomi * floats_per_node;
        const IN_T* input_j = input + atomj * floats_per_node;
        // For each edge, record all of the sums followed by all of the absolute differences
        OUT_T* output_sum = output;
        OUT_T* output_absdiff = output + floats_per_node;
        for (size_t float_index = 0; float_index < floats_per_node; ++float_index) {
            const IN_T sum = input_i[float_index] + input_j[float_index];
            const IN_T diff = input_i[float_index] - input_j[float_index];
            const IN_T absdiff = std::abs(diff);
            const OUT_T sum_out = FeatureValues<OUT_T>::convertToFeatureType(sum);
            const OUT_T absdiff_out = FeatureValues<OUT_T>::convertToFeatureType(absdiff);
            output_sum[float_index] = sum_out;
            output_absdiff[float_index] = absdiff_out;
            // Same values for opposite direction
            output_sum[floats_per_half_edge + float_index] = sum_out;
            output_absdiff[floats_per_half_edge + float_index] = absdiff_out;
        }
    }
}

template<typename OUT_T, typename IN_T>
void node_to_node_pair(
    std::unique_ptr<OUT_T[]>& output_ptr,
    size_t& floats_per_pair,
    const IN_T* input,
    const size_t n,
    const size_t floats_per_node) {

    floats_per_pair = 2 * floats_per_node;
    output_ptr.reset(new OUT_T[n * n * floats_per_pair]);
    OUT_T* output = output_ptr.get();
    const IN_T* input_i = input;
    for (size_t i = 0; i < n; ++i, input_i += floats_per_node) {
        const IN_T* input_j = input;
        for (size_t j = 0; j < n; ++j, input_j += floats_per_node, output += floats_per_pair) {
            // For each pair, record all of the sums followed by all of the absolute differences
            OUT_T* output_sum = output;
            OUT_T* output_absdiff = output + floats_per_node;
            for (size_t float_index = 0; float_index < floats_per_node; ++float_index) {
                const IN_T sum = input_i[float_index] + input_j[float_index];
                const IN_T diff = input_i[float_index] - input_j[float_index];
                const IN_T absdiff = std::abs(diff);
                output_sum[float_index] = FeatureValues<OUT_T>::convertToFeatureType(sum);
                output_absdiff[float_index] = FeatureValues<OUT_T>::convertToFeatureType(absdiff);
            }
        }
    }
}

enum class StatOperation {
    MINIMUM,
    MEAN
};

template<StatOperation op, typename T>
T stat_init_accum(T v) {
    return v;
}

template<StatOperation op, typename T>
void stat_accum(T& accum, T v) {
    switch (op) {
    case StatOperation::MINIMUM:
        accum = (v < accum) ? v : accum;
        break;
    case StatOperation::MEAN:
        accum += v;
        break;
    }
}

template<StatOperation op, typename T>
T stat_accum_finish(T accum, size_t n) {
    switch (op) {
    case StatOperation::MINIMUM:
        return accum;
    case StatOperation::MEAN:
        return accum / n;
    }
}

template<StatOperation op, typename OUT_T, typename IN_T>
void node_pair_to_node_helper(
    OUT_T* output,
    const IN_T* input,
    const size_t n,
    const size_t floats_per_pair,
    const size_t node_index) {

    // for each float per pair
    for (size_t float_index = 0; float_index < floats_per_pair; ++float_index, output += 2) {
        // across all rows (axis 0) of column node_index, then across all columns (axis 1) of row node_index
        IN_T accum = stat_init_accum<op>(input[node_index * floats_per_pair + float_index]);
        for (size_t row = 1; row < n; ++row) {
            stat_accum<op>(accum, input[(row * n + node_index) * floats_per_pair + float_index]);
        }
        output[0] = FeatureValues<OUT_T>::convertToFeatureType(stat_accum_finish<op>(accum, n));
        accum = stat_init_accum<op>(input[node_index * n * floats_per_pair + float_index]);
        for (size_t col = 1; col < n; ++col) {
            stat_accum<op>(accum, input[(node_index * n + col) * floats_per_pair + float_index]);
        }
        output[1] = FeatureValues<OUT_T>::convertToFeatureType(stat_accum_finish<op>(accum, n));
    }
}

template<typename OUT_T, typename IN_T>
void node_pair_to_node_helper_stdev(
    OUT_T* output,
    const IN_T* input,
    const size_t n,
    const size_t floats_per_pair,
    const size_t node_index) {

    // for each float per pair
    for (size_t float_index = 0; float_index < floats_per_pair; ++float_index, output += 2) {
        // across all rows (axis 0) of column node_index, then across all columns (axis 1) of row node_index
        IN_T v = input[node_index * floats_per_pair + float_index];
        IN_T accum = v;
        IN_T accum2 = v * v;
        for (size_t row = 1; row < n; ++row) {
            v = input[(row * n + node_index) * floats_per_pair + float_index];
            accum += v;
            accum2 += v * v;
        }
        // NOTE: Using divisor n, the default in numpy.std, not n-1, the default elsewhere
        accum /= n;
        accum2 /= n;
        output[0] = FeatureValues<OUT_T>::convertToFeatureType(std::sqrt(accum2 - accum*accum));

        v = input[node_index * n * floats_per_pair + float_index];
        accum = v;
        accum2 = v * v;
        for (size_t col = 1; col < n; ++col) {
            v = input[(node_index * n + col) * floats_per_pair + float_index];
            accum += v;
            accum2 += v * v;
        }
        // NOTE: Using divisor n, the default in numpy.std, not n-1, the default elsewhere
        accum /= n;
        accum2 /= n;
        output[1] = FeatureValues<OUT_T>::convertToFeatureType(std::sqrt(accum2 - accum*accum));
    }
}

template<typename OUT_T, typename IN_T>
void node_pair_to_node(
    std::unique_ptr<OUT_T[]>& output_ptr,
    size_t& floats_per_node,
    const IN_T* input,
    const size_t n,
    const size_t floats_per_pair) {

    const size_t num_ops = 3;
    floats_per_node = num_ops * 2 * floats_per_pair;
    output_ptr.reset(new OUT_T[n * floats_per_node]);
    OUT_T* output = output_ptr.get();
    for (size_t node_index = 0; node_index < n; ++node_index) {
        // min, mean, stdev (using divisor N, the default in numpy.std, not N-1, the default elsewhere)
        node_pair_to_node_helper<StatOperation::MINIMUM>(output, input, n, floats_per_pair, node_index);
        output += 2 * floats_per_pair;
        node_pair_to_node_helper<StatOperation::MEAN>(output, input, n, floats_per_pair, node_index);
        output += 2 * floats_per_pair;
        node_pair_to_node_helper_stdev(output, input, n, floats_per_pair, node_index);
        output += 2 * floats_per_pair;
    }
}

template<typename OUT_T, typename IN_T>
void node_pair_to_edge(
    std::unique_ptr<OUT_T[]>& output_ptr,
    size_t& floats_per_edge,
    const IN_T* input,
    const size_t n,
    const size_t floats_per_pair,
    const GraphData& graph) {

    // Edge order must be consistent with the edges in the graph,
    // which is not necessarily lexicographic order.
    const size_t num_half_edges = 2*graph.num_bonds;
    floats_per_edge = floats_per_pair;
    output_ptr.reset(new OUT_T[num_half_edges * floats_per_pair]);
    OUT_T* output = output_ptr.get();
    for (size_t bond = 0; bond < graph.num_bonds; ++bond) {
        const size_t atomi = graph.bonds[bond].beginAtomIdx;
        const size_t atomj = graph.bonds[bond].endAtomIdx;
        const IN_T* input_ij = input + ((atomi * n) + atomj) * floats_per_pair;
        for (size_t float_index = 0; float_index < floats_per_pair; ++float_index, ++output) {
            *output = FeatureValues<OUT_T>::convertToFeatureType(input_ij[float_index]);
        }
        
        const IN_T* input_ji = input + ((atomj * n) + atomi) * floats_per_pair;
        for (size_t float_index = 0; float_index < floats_per_pair; ++float_index, ++output) {
            *output = FeatureValues<OUT_T>::convertToFeatureType(input_ji[float_index]);
        }
    }
}

template<typename T>
void create_positional_features(
    const GraphData& graph,
    const at::Tensor& positional_property_list,
    c10::ScalarType dtype,
    MaskNaNStyle mask_nan_style,
    T mask_nan_value,
    int64_t& num_nans,
    int64_t& nan_tensor_index,
    std::vector<at::Tensor>& tensors) {

    const size_t size = (positional_property_list.scalar_type() == c10::ScalarType::Long && positional_property_list.ndimension() == 1) ? positional_property_list.size(0) : 0;
    const int64_t* const property_list = (size >= 3) ? positional_property_list.data_ptr<int64_t>() : nullptr;

    if (property_list == nullptr) {
        return;
    }
    NeighbourData neighbours = construct_neighbours(graph);

    LaplacianData<double> laplacian_data;
    LaplacianData<double> laplacian_data_comp;
    size_t num_components = 0; // 0 indicates that the components haven't been computed yet
    std::vector<int32_t> components;
    std::vector<double> laplacian_pseudoinverse;
    std::vector<double> matrix;
    size_t i = 0;
    while (size >= i + 3) {
        int64_t property = property_list[i];
        int64_t current_size = property_list[i + 1];
        FeatureLevel feature_level = FeatureLevel(property_list[i + 2]);
        i += 3;
        if (i + current_size > size || i + current_size < i) {
            break;
        }
        FeatureLevel base_level;
        std::unique_ptr<double[]> base_data;
        int64_t base_dims[3] = { 1,1,1 };
        size_t base_dim_count;
        if ((property == int64_t(PositionalFeature::LAPLACIAN_EIGENVEC) || property == int64_t(PositionalFeature::LAPLACIAN_EIGENVAL)) && current_size == 3) {
            size_t num_pos = (property_list[i] >= 0) ? size_t(property_list[i]) : 0;
            Normalization normalization = Normalization(property_list[i + 1]);
            bool disconnected_comp = (property_list[i + 2] != 0);
            i += 3;

            // The common case is that there's only 1 component, even if disconnected_comp is true,
            // so find the number of components, first.
            if (disconnected_comp && num_components == 0) {
                num_components = find_components(graph.num_atoms, neighbours.neighbour_starts, neighbours.neighbours, components);
            }
            const bool multiple_components = disconnected_comp && (num_components > 1);

            LaplacianData<double>& current_data = multiple_components ? laplacian_data_comp : laplacian_data;
            if (current_data.eigenvalues.size() == 0 || current_data.normalization != normalization) {
                compute_laplacian_eigendecomp(
                    graph.num_atoms,
                    neighbours.neighbour_starts,
                    neighbours.neighbours,
                    normalization,
                    current_data,
                    multiple_components ? num_components : 1,
                    components.data());
            }

            const bool isVec = (property == int64_t(PositionalFeature::LAPLACIAN_EIGENVEC));
            base_level = FeatureLevel::NODE;
            base_dims[0] = graph.num_atoms;
            base_dims[1] = num_pos;
            base_dim_count = 2;
            base_data.reset(new double[graph.num_atoms * num_pos]);

            // Ensure exactly the tensor dimensions of num_atoms x num_pos before changing the level.
            if (isVec) {
                double* data = base_data.get();
                for (size_t atom_index = 0; atom_index < graph.num_atoms; ++atom_index, data += num_pos) {
                    for (size_t i = 0; i < num_pos && i < graph.num_atoms; ++i) {
                        // Row eigenvectors to column eigenvectors
                        data[i] = current_data.vectors[atom_index + i * graph.num_atoms];
                        // There's no plausible way the eigenvectors should end up with NaNs,
                        // so just assert in debug builds.
                        assert(std::isfinite(data[i]));
                    }
                    // NOTE: Do not treat extra values as NaN.  The original code filled them with zeros.
                    for (size_t i = graph.num_atoms; i < num_pos; ++i) {
                        data[i] = 0;
                    }
                }
            }
            else {
                double* data = base_data.get();
                const bool is_multi_component = (current_data.eigenvalues.size() == size_t(graph.num_atoms)*graph.num_atoms);
                assert(is_multi_component || (current_data.eigenvalues.size() == graph.num_atoms));
                size_t source_row_start = 0;
                for (size_t atom_index = 0; atom_index < graph.num_atoms; ++atom_index, data += num_pos) {
                    for (size_t i = 0; i < num_pos && i < graph.num_atoms; ++i) {
                        // Duplicate the eigenvalue for each atom
                        data[i] = current_data.eigenvalues[source_row_start + i];
                        // There's no plausible way the eigenvalues should end up with NaNs,
                        // so just assert in debug builds.
                        assert(std::isfinite(data[i]));
                    }
                    // NOTE: Do not treat extra values as NaN.  The original code filled them with zeros.
                    for (size_t i = graph.num_atoms; i < num_pos; ++i) {
                        data[i] = 0;
                    }
                    if (is_multi_component) {
                        source_row_start += graph.num_atoms;
                    }
                }
            }
        }
        else if ((property == int64_t(PositionalFeature::RW_RETURN_PROBS) || property == int64_t(PositionalFeature::RW_TRANSITION_PROBS)) && current_size >= 1) {
            int space_dim = property_list[i];
            ++i;
            uint32_t num_powers = current_size - 1;
            const uint64_t* powers = reinterpret_cast<const uint64_t*>(property_list + i);
            i += num_powers;
            const bool isProbs = (property == int64_t(PositionalFeature::RW_RETURN_PROBS));
            RandomWalkDataOption option = isProbs ? RandomWalkDataOption::PROBABILITIES : RandomWalkDataOption::MATRIX;

            std::vector<double> output;
            compute_rwse(num_powers, powers, graph.num_atoms, neighbours.neighbour_starts, neighbours.neighbours, option, output, space_dim);

            base_level = isProbs ? FeatureLevel::NODE : FeatureLevel::NODEPAIR;

            base_dims[0] = graph.num_atoms;
            base_dims[1] = isProbs ? num_powers : graph.num_atoms;
            base_dims[2] = isProbs ? 1 : num_powers;
            base_dim_count = isProbs ? 2 : 3;
            base_data.reset(new double[output.size()]);
            std::copy(output.begin(), output.end(), base_data.get());
        }
        else if (property == int64_t(PositionalFeature::ELECTROSTATIC) && current_size == 0) {
            const double* weights = nullptr;
            compute_electrostatic_interactions(graph.num_atoms, neighbours.neighbour_starts, neighbours.neighbours, laplacian_data, laplacian_pseudoinverse, matrix, weights);

            base_level = FeatureLevel::NODEPAIR;
            base_dims[0] = graph.num_atoms;
            base_dims[1] = graph.num_atoms;
            base_dim_count = 2;
            assert(matrix.size() == graph.num_atoms * size_t(graph.num_atoms));
            base_data.reset(new double[matrix.size()]);
            std::copy(matrix.begin(), matrix.end(), base_data.get());
        }
        else if (property == int64_t(PositionalFeature::COMMUTE) && current_size == 0) {
            const double* weights = nullptr;
            compute_commute_distances(graph.num_atoms, neighbours.neighbour_starts, neighbours.neighbours, laplacian_data, laplacian_pseudoinverse, matrix, weights);

            base_level = FeatureLevel::NODEPAIR;
            base_dims[0] = graph.num_atoms;
            base_dims[1] = graph.num_atoms;
            base_dim_count = 2;
            assert(matrix.size() == graph.num_atoms * size_t(graph.num_atoms));
            base_data.reset(new double[matrix.size()]);
            std::copy(matrix.begin(), matrix.end(), base_data.get());
        }
        else if (property == int64_t(PositionalFeature::GRAPHORMER) && current_size == 0) {
            std::vector<std::pair<uint32_t, uint32_t>> queue;
            std::vector<double> all_pairs_distances;
            compute_graphormer_distances(graph.num_atoms, neighbours.neighbour_starts, neighbours.neighbours, queue, all_pairs_distances);

            base_level = FeatureLevel::NODEPAIR;
            base_dims[0] = graph.num_atoms;
            base_dims[1] = graph.num_atoms;
            base_dim_count = 2;
            assert(all_pairs_distances.size() == graph.num_atoms * size_t(graph.num_atoms));
            base_data.reset(new double[all_pairs_distances.size()]);
            std::copy(all_pairs_distances.begin(), all_pairs_distances.end(), base_data.get());
        }

        if (base_data.get() == nullptr) {
            continue;
        }

        // Change the level and convert to the correct type if needed.
        std::unique_ptr<T[]> final_data;
        int64_t final_dims[3];
        std::copy(base_dims, base_dims + 3, final_dims);
        size_t final_num_dims = base_dim_count;
        if (feature_level != base_level) {
            if (base_level == FeatureLevel::NODE) {
                if (feature_level == FeatureLevel::EDGE) {
                    size_t floats_per_half_edge;
                    node_to_edge(final_data, floats_per_half_edge, base_data.get(), base_dims[0], base_dims[1], graph);
                    final_dims[0] = 2 * graph.num_bonds;
                    final_dims[1] = floats_per_half_edge;
                    final_dims[2] = 1;
                }
                else if (feature_level == FeatureLevel::NODEPAIR) {
                    size_t floats_per_pair;
                    node_to_node_pair(final_data, floats_per_pair, base_data.get(), base_dims[0], base_dims[1]);
                    final_num_dims = 3;
                    final_dims[1] = base_dims[0];
                    final_dims[2] = floats_per_pair;
                }
                else {
                    // Not implemented
                }
            }
            else if (base_level == FeatureLevel::NODEPAIR) {
                if (feature_level == FeatureLevel::NODE) {
                    size_t floats_per_node;
                    node_pair_to_node(final_data, floats_per_node, base_data.get(), base_dims[0], base_dims[2]);
                    final_num_dims = 2;
                    final_dims[1] = floats_per_node;
                    final_dims[2] = 1;
                }
                else if (feature_level == FeatureLevel::EDGE) {
                    size_t floats_per_edge;
                    node_pair_to_edge(final_data, floats_per_edge, base_data.get(), base_dims[0], base_dims[2], graph);
                    final_num_dims = 2;
                    final_dims[0] = 2 * graph.num_bonds;
                    final_dims[1] = floats_per_edge;
                    final_dims[2] = 1;
                }
                else {
                    // Not implemented
                }
            }
            else {
                // Not implemented
            }
        }
        else if (dtype != c10::ScalarType::Double) {
            // Just convert
            const size_t total_num_floats = final_dims[0] * final_dims[1] * final_dims[2];
            final_data.reset(new T[total_num_floats]);
            for (size_t i = 0; i < total_num_floats; ++i) {
                final_data[i] = FeatureValues<T>::convertToFeatureType(base_data[i]);
            }
        }
        else {
            // Perfect match out of the box
            // This will only be hit if T is double, but it still needs to compile
            // for other cases, which is why the reinterpret_cast is needed.
            final_data.reset(reinterpret_cast<T*>(base_data.release()));
        }

        if (final_data.get() == nullptr) {
            continue;
        }

        tensors.push_back(torch_tensor_from_array<T>(std::move(final_data), final_dims, final_num_dims, dtype));
    }
}

template<typename T>
void create_all_features(
    const GraphData& graph,
    const at::Tensor& atom_property_list_onehot,
    const at::Tensor& atom_property_list_float,
    bool create_conformer_feature,
    const at::Tensor& bond_property_list,
    const at::Tensor& positional_property_list,
    bool duplicate_edges,
    bool add_self_loop,
    bool already_has_Hs,
    bool use_bonds_weights,
    bool offset_carbon,
    c10::ScalarType dtype,
    MaskNaNStyle mask_nan_style,
    T mask_nan_value,
    int64_t& num_nans,
    int64_t& nan_tensor_index,
    const std::string& smiles_string,
    std::vector<at::Tensor>& tensors) {

    if (mask_nan_style == MaskNaNStyle::NONE) {
        // In some cases, the NONE and REPLACE styles can be combined.
        mask_nan_value = FeatureValues<T>::nan_value;
    }
    at::Tensor edge_weights_tensor = create_edge_weights<T>(
        graph,
        duplicate_edges,
        add_self_loop,
        use_bonds_weights,
        dtype);
    tensors.push_back(std::move(edge_weights_tensor));
    at::Tensor atom_features_tensor = create_atom_features<T>(
        graph,
        atom_property_list_onehot,
        atom_property_list_float,
        offset_carbon,
        dtype,
        mask_nan_style,
        mask_nan_value,
        num_nans);
    tensors.push_back(std::move(atom_features_tensor));
    if (num_nans != 0) {
        nan_tensor_index = tensors.size()-1;
    }
    at::Tensor bond_features_tensor = create_bond_features<T>(
        graph,
        bond_property_list,
        duplicate_edges,
        add_self_loop,
        dtype,
        mask_nan_style,
        mask_nan_value,
        num_nans);
    tensors.push_back(std::move(bond_features_tensor));
    if (nan_tensor_index < 0 && num_nans != 0) {
        nan_tensor_index = tensors.size()-1;
    }
    if (create_conformer_feature) {
        at::Tensor conformer_features_tensor = get_conformer_features<T>(
            *graph.mol,
            already_has_Hs,
            dtype,
            mask_nan_style,
            mask_nan_value,
            num_nans,
            smiles_string);
        tensors.push_back(std::move(conformer_features_tensor));
        if (nan_tensor_index < 0 && num_nans != 0) {
            nan_tensor_index = tensors.size();
        }
    }
    create_positional_features<T>(
        graph,
        positional_property_list,
        dtype,
        mask_nan_style,
        mask_nan_value,
        num_nans,
        nan_tensor_index,
        tensors);
}

std::tuple<std::vector<at::Tensor>, int64_t, int64_t> featurize_smiles(
    const std::string& smiles_string,
    const at::Tensor& atom_property_list_onehot,
    const at::Tensor& atom_property_list_float,
    bool create_conformer_feature,
    const at::Tensor& bond_property_list,
    const at::Tensor& positional_property_list,
    bool duplicate_edges,
    bool add_self_loop,
    bool explicit_H,
    bool use_bonds_weights,
    bool offset_carbon,
    int dtype_int,
    int mask_nan_style_int,
    double mask_nan_value) {

    GraphData graph = read_graph(smiles_string, explicit_H);

    const size_t edge_coo_count = 2*graph.num_bonds + (add_self_loop ? graph.num_atoms : 0);
    std::unique_ptr<int64_t[]> edge_index(new int64_t[2*edge_coo_count]);
    for (size_t i = 0; i < graph.num_bonds; ++i) {
        // PyG has all directed edge begin indices followed by all end indices.
        edge_index[2*i] = graph.bonds[i].beginAtomIdx;
        edge_index[2*i+1] = graph.bonds[i].endAtomIdx;
        edge_index[2*i + edge_coo_count] = graph.bonds[i].endAtomIdx;
        edge_index[2*i+1 + edge_coo_count] = graph.bonds[i].beginAtomIdx;
    }
    if (add_self_loop) {
        for (size_t i = 0; i < graph.num_atoms; ++i) {
            edge_index[2*graph.num_bonds + i] = i;
            edge_index[2*graph.num_bonds + i + edge_coo_count] = i;
        }
    }
    int64_t edge_coo_dims[2] = { int64_t(2), int64_t(edge_coo_count) };
    at::Tensor edge_coo_tensor = torch_tensor_from_array<int64_t>(std::move(edge_index), edge_coo_dims, 2, c10::ScalarType::Long);

    std::vector<at::Tensor> tensors;
    tensors.push_back(std::move(edge_coo_tensor));
    c10::ScalarType dtype = c10::ScalarType(dtype_int);
    MaskNaNStyle mask_nan_style = MaskNaNStyle(mask_nan_style_int);
    int64_t num_nans = 0;
    int64_t nan_tensor_index = -1;
    if (dtype == c10::ScalarType::Half) {
        create_all_features<int16_t>(
            graph,
            atom_property_list_onehot,
            atom_property_list_float,
            create_conformer_feature,
            bond_property_list,
            positional_property_list,
            duplicate_edges,
            add_self_loop,
            explicit_H,
            use_bonds_weights,
            offset_carbon,
            dtype,
            mask_nan_style,
            FeatureValues<int16_t>::convertToFeatureType(mask_nan_value),
            num_nans,
            nan_tensor_index,
            smiles_string,
            tensors);
    }
    else if (dtype == c10::ScalarType::Float) {
        create_all_features<float>(
            graph,
            atom_property_list_onehot,
            atom_property_list_float,
            create_conformer_feature,
            bond_property_list,
            positional_property_list,
            duplicate_edges,
            add_self_loop,
            explicit_H,
            use_bonds_weights,
            offset_carbon,
            dtype,
            mask_nan_style,
            FeatureValues<float>::convertToFeatureType(mask_nan_value),
            num_nans,
            nan_tensor_index,
            smiles_string,
            tensors);
    }
    else if (dtype == c10::ScalarType::Double) {
        create_all_features<double>(
            graph,
            atom_property_list_onehot,
            atom_property_list_float,
            create_conformer_feature,
            bond_property_list,
            positional_property_list,
            duplicate_edges,
            add_self_loop,
            explicit_H,
            use_bonds_weights,
            offset_carbon,
            dtype,
            mask_nan_style,
            FeatureValues<double>::convertToFeatureType(mask_nan_value),
            num_nans,
            nan_tensor_index,
            smiles_string,
            tensors);
    }

    return std::make_tuple(tensors, num_nans, nan_tensor_index);
}
