"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


"""
Unit tests for the positional encodings in graphium/features/*
"""

import numpy as np
import torch
import unittest as ut

import graphium
import graphium_cpp

class test_positional_encodings(ut.TestCase):
    # Test graphs
    smiles_dict = {}
    shape_dict = {}
    max_dict = {}

    # 6-ring
    smiles = "C1CCCCC1"
    smiles_dict["6-ring"] = smiles
    shape_dict["6-ring"] = [6, 6]
    max_dict["6-ring"] = 3

    # 5-path
    smiles = "CCCCC"
    smiles_dict["5-path"] = smiles
    shape_dict["5-path"] = [5, 5]
    max_dict["5-path"] = 4

    # 4-clique
    smiles = "C12C3C1C23"
    smiles_dict["4-clique"] = smiles
    shape_dict["4-clique"] = [4, 4]
    max_dict["4-clique"] = 1

    # 4-barbell
    smiles = "C12C3C1C23C12C3C1C23"
    smiles_dict["4-barbell"] = smiles
    shape_dict["4-barbell"] = [8, 8]
    max_dict["4-barbell"] = 3

    features = {
        "electrostatic": {"pos_level": "nodepair", "pos_type": "electrostatic", "normalization": "none"},
        "graphormer": {"pos_level": "nodepair", "pos_type": "graphormer", "normalization": "none"},
        "commute": {"pos_level": "nodepair", "pos_type": "commute", "normalization": "none"},
        }
    (pos_encoding_names, pos_encoding_tensor) = \
                graphium_cpp.positional_feature_options_to_tensor(features)

    def get_tensors(self, smiles):
        tensors, _, _ = graphium_cpp.featurize_smiles(
            smiles,
            torch.tensor(data=[], dtype=torch.int64), # atom_property_list_onehot
            torch.tensor(data=[], dtype=torch.int64), # atom_property_list_float
            False, # has_conformer
            torch.tensor(data=[], dtype=torch.int64), # edge_property_list
            self.pos_encoding_tensor,
            True, # duplicate_edges
            False, # add_self_loop
            False, # explicit_H=False
            False, # use_bonds_weights
            True, #offset_carbon
            7, # torch float64
            0, # mask_nan_style_int
            0  # mask_nan_value
        )
        return tensors

    def test_dimensions(self):
        for key, smiles in self.smiles_dict.items():
            tensors = self.get_tensors(smiles)

            pe = tensors[4] # electrostatic
            self.assertEqual(list(pe.shape), self.shape_dict[key])

            pe = tensors[5] # graphormer
            self.assertEqual(list(pe.shape), self.shape_dict[key])

            pe = tensors[6] # commute
            self.assertEqual(list(pe.shape), self.shape_dict[key])

    def test_symmetry(self):
        for _, smiles in self.smiles_dict.items():
            tensors = self.get_tensors(smiles)

            pe = tensors[5] # graphormer
            np.testing.assert_array_almost_equal(pe, pe.T)

            pe = tensors[6] # commute
            np.testing.assert_array_almost_equal(pe, pe.T)

    def test_max_dist(self):
        for key, smiles in self.smiles_dict.items():
            tensors = self.get_tensors(smiles)

            pe = tensors[5] # graphormer
            np.testing.assert_array_almost_equal(pe.max(), self.max_dict[key])


if __name__ == "__main__":
    ut.main()
