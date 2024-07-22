// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "features.h"
#include "labels.h"

// C++ standard library headers
#include <assert.h>
#include <filesystem>
#include <memory>
#include <numeric>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <utility>
#include <vector>

// RDKit headers
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/ROMol.h>
#include <GraphMol/RWMol.h>
#include <GraphMol/Canon.h>
#include <GraphMol/new_canon.h>
#include <GraphMol/MolOps.h>

// PyBind and Torch headers for use by library to be imported by Python
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// RDKit::SmilesToMol uses std::string, so until we replace it, lets use std::string here.
// ("const char*" could avoid an extra allocation, if we do eventually replace use of SmilesToMol.)
std::unique_ptr<RDKit::RWMol> parse_mol(
    const std::string& smiles_string,
    bool explicit_H,
    bool ordered) {

    // Parse SMILES string with default options
    RDKit::SmilesParserParams params;
    std::unique_ptr<RDKit::RWMol> mol{ RDKit::SmilesToMol(smiles_string, params) };
    if (!mol) {
        return mol;
    }

    if (ordered) {
        // Determine a canonical ordering of the atoms
        const unsigned int num_atoms = mol->getNumAtoms();
        std::vector<unsigned int> atom_order;
        RDKit::Canon::rankMolAtoms(*mol, atom_order);
        assert(atom_order.size() == num_atoms);

        // Invert the order
        std::vector<unsigned int> inverse_order(num_atoms);
        for (unsigned int i = 0; i < num_atoms; ++i) {
            inverse_order[atom_order[i]] = i;
        }
        
        // Reorder the atoms to the canonical order
        mol.reset(static_cast<RDKit::RWMol*>(RDKit::MolOps::renumberAtoms(*mol, inverse_order)));
    }
    if (explicit_H) {
        RDKit::MolOps::addHs(*mol);
    }
    else {
        // Default params for SmilesToMol already calls removeHs,
        // and calling it again shouldn't have any net effect.
        //RDKit::MolOps::removeHs(*mol);
    }
    return mol;
}

void get_canonical_atom_order(const RDKit::ROMol& mol, std::vector<unsigned int>& atom_order) {
    RDKit::Canon::rankMolAtoms(mol, atom_order);
    assert(atom_order.size() == mol->getNumAtoms());
}

// This is necessary to export Python functions in a Python module named graphium_cpp.
PYBIND11_MODULE(graphium_cpp, m) {
    m.doc() = "graphium C++ plugin"; // Python module docstring

    // Functions in labels.cpp
    m.def("load_num_cols_and_dtypes", &load_num_cols_and_dtypes, "Loads from a cache file, a list of integers representing the number of columns in each task, and a list of integers representing the torch ScalarType of the task's data.");
    m.def("load_metadata_tensors", &load_metadata_tensors, "Loads from cache files for a specific stage, a torch tensor containing all SMILES strings contatenated, another with the offsets of all SMILES strings, two for the nubmer of nodes and edges in each molecule, and optionally another representing the offsets of molecules in files.");
    m.def("load_stats", &load_stats, "Loads from a cache file of a specific task, the stats for each column, for use in denormalization.");
    m.def("concatenate_strings", &concatenate_strings, "Accepts a Numpy array of strings or Python list of strings and returns a PyTorch tensor of all of the characters and another tensor containing indices into the other tensor indicating where each string begins.");
    m.def("prepare_and_save_data", &prepare_and_save_data, "Accepts a dict mapping dataset (task) names to dicts with \"smiles\", \"labels\", and \"label_offsets\" data, and returns the data that would be returned by load_metadata_tensors, load_stats, and load_num_cols_and_dtypes.");
    m.def("load_labels_from_index", &load_labels_from_index, "Loads label data from disk, for a specific stage and molecule.");
    m.def("extract_string", &extract_string, "Extracts a single string from a Tensor of contatenated strings.");

    // Functions in features.cpp
    m.def("atom_float_feature_names_to_tensor", &atom_float_feature_names_to_tensor, "Accepts feature names and returns a tensor representing them as integers");
    m.def("atom_onehot_feature_names_to_tensor", &atom_onehot_feature_names_to_tensor, "Accepts feature names and returns a tensor representing them as integers");
    m.def("bond_feature_names_to_tensor", &bond_feature_names_to_tensor, "Accepts feature names and returns a tensor representing them as integers");
    m.def("positional_feature_options_to_tensor", &positional_feature_options_to_tensor, "Accepts feature names, levels, and options, and returns a tensor representing them as integers");
    m.def("featurize_smiles", &featurize_smiles, "Accepts a SMILES string and returns tensors representing the features");
}
