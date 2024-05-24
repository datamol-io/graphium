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
Unit tests for the positional encodings in graphium/features/random_walk.cpp
"""

import numpy as np
import torch
import unittest as ut

import graphium
import graphium_cpp

class test_pe_spectral(ut.TestCase):
    def test_outputs(self):
        # 4-barbell
        smiles = "C12C3C1C23C12C3C1C23"
        num_nodes = 8

        ksteps1 = [4, 6]
        ksteps2 = [2]
        ksteps3 = [6, 7]

        # The feature names only depend on pos_type and pos_level, so the two
        # rw_return_probs features can't have the same pos_level.
        features = {
            "rw_transition_probs": {"pos_level": "nodepair", "pos_type": "rw_transition_probs", "normalization": "none", "ksteps": ksteps1},
            "rw_return_probs_0": {"pos_level": "node", "pos_type": "rw_return_probs", "normalization": "none", "ksteps": ksteps2},
            "rw_return_probs_1": {"pos_level": "nodepair", "pos_type": "rw_return_probs", "normalization": "none", "ksteps": ksteps3},
            }
        (pos_encoding_names, pos_encoding_tensor) = \
                    graphium_cpp.positional_feature_options_to_tensor(features)

        tensors, _, _ = graphium_cpp.featurize_smiles(
            smiles,
            torch.tensor(data=[], dtype=torch.int64), # atom_property_list_onehot
            torch.tensor(data=[], dtype=torch.int64), # atom_property_list_float
            False, # has_conformer
            torch.tensor(data=[], dtype=torch.int64), # edge_property_list
            pos_encoding_tensor,
            True, # duplicate_edges
            False, # add_self_loop
            False, # explicit_H=False
            False, # use_bonds_weights
            True, #offset_carbon
            7, # torch float64
            0, # mask_nan_style_int
            0  # mask_nan_value
        )

        pe1 = tensors[4]
        pe2 = tensors[5]
        pe3 = tensors[6]

        self.assertTrue(pe1.shape, np.zeros((num_nodes, num_nodes, len(ksteps1))))
        self.assertTrue(pe2.shape, np.zeros((num_nodes, len(ksteps2))))
        self.assertTrue(pe3.shape, np.zeros((num_nodes, len(ksteps3))))

        for i in range(len(ksteps1)):
            np.testing.assert_array_almost_equal(pe1[..., i].sum(1), np.ones(num_nodes))

        self.assertGreaterEqual(pe1.min(), 0.0)
        self.assertLessEqual(pe1.max(), 1.0)

        self.assertGreaterEqual(pe2.min(), 0.0)
        self.assertLessEqual(pe2.max(), 1.0)

        self.assertGreaterEqual(pe3.min(), 0.0)
        self.assertLessEqual(pe3.max(), 1.0)


if __name__ == "__main__":
    ut.main()
