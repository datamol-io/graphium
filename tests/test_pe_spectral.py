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


def get_pe_tensors(smiles, pos_encoding_tensor):
    tensors, _, _ = graphium_cpp.featurize_smiles(
        smiles,
        torch.tensor(data=[], dtype=torch.int64),  # atom_property_list_onehot
        torch.tensor(data=[], dtype=torch.int64),  # atom_property_list_float
        False,  # has_conformer
        torch.tensor(data=[], dtype=torch.int64),  # edge_property_list
        pos_encoding_tensor,
        True,  # duplicate_edges
        False,  # add_self_loop
        False,  # explicit_H=False
        False,  # use_bonds_weights
        True,  # offset_carbon
        7,  # torch float64
        0,  # mask_nan_style_int
        0,  # mask_nan_value
    )
    return tensors


class test_pe_spectral(ut.TestCase):
    def test_for_connected_vs_disconnected_graph(self):
        # 2 disconnected 3 cliques
        smiles1 = "C1CC1.C1CC1"

        # 6-clique (have to use S instead of C, because RDKit doesn't accept a carbon having 6 explicit bonds)
        smiles2 = "S1234S567S189S251S368S4791"

        num_atoms = 6
        num_pos = 3

        features = {
            "laplacian_eigval": {
                "pos_level": "node",
                "pos_type": "laplacian_eigval",
                "normalization": "none",
                "num_pos": num_pos,
                "disconnected_comp": True,
            },
            "laplacian_eigvec": {
                "pos_level": "node",
                "pos_type": "laplacian_eigvec",
                "normalization": "none",
                "num_pos": num_pos,
                "disconnected_comp": True,
            },
        }
        (pos_encoding_names, pos_encoding_tensor) = graphium_cpp.positional_feature_options_to_tensor(
            features
        )

        # test if pe works identically on connected vs disconnected graphs
        tensors1 = get_pe_tensors(smiles1, pos_encoding_tensor)
        eigvals_pe1 = tensors1[4]
        eigvecs_pe1 = tensors1[5]
        tensors2 = get_pe_tensors(smiles2, pos_encoding_tensor)
        eigvals_pe2 = tensors2[4]
        eigvecs_pe2 = tensors2[5]

        np.testing.assert_array_almost_equal(2 * eigvals_pe1, eigvals_pe2)
        self.assertListEqual(list(eigvals_pe2.shape), [num_atoms, num_pos])
        self.assertListEqual(list(eigvecs_pe2.shape), [num_atoms, num_pos])


if __name__ == "__main__":
    ut.main()
