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
import math

import graphium
import graphium_cpp


def get_tensors(smiles, pos_encoding_tensor):
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


class test_pos_transfer_funcs(ut.TestCase):

    def test_different_transfers(self):
        smiles = "CCCC"

        ksteps = [2, 4]
        features = {
            "a": {
                "pos_level": "node",
                "pos_type": "rw_return_probs",
                "normalization": "none",
                "ksteps": ksteps,
            },
            "b": {
                "pos_level": "edge",
                "pos_type": "rw_return_probs",
                "normalization": "none",
                "ksteps": ksteps,
            },
            "c": {
                "pos_level": "nodepair",
                "pos_type": "rw_return_probs",
                "normalization": "none",
                "ksteps": ksteps,
            },
            "e": {"pos_level": "node", "pos_type": "graphormer", "normalization": "none"},
            "f": {"pos_level": "edge", "pos_type": "graphormer", "normalization": "none"},
            "d": {"pos_level": "nodepair", "pos_type": "graphormer", "normalization": "none"},
        }

        (pos_encoding_names, pos_encoding_tensor) = graphium_cpp.positional_feature_options_to_tensor(
            features
        )

        tensors = get_tensors(smiles, pos_encoding_tensor)
        node_probs = tensors[4]
        edge_probs = tensors[5]
        nodepair_probs = tensors[6]
        node_dists = tensors[7]
        edge_dists = tensors[8]
        nodepair_dists = tensors[9]

        print(f"node_probs =\n{node_probs}\n")
        print(f"edge_probs =\n{edge_probs}\n")
        print(f"nodepair_probs =\n{nodepair_probs}\n")
        print(f"node_dists =\n{node_dists}\n")
        print(f"edge_dists =\n{edge_dists}\n")
        print(f"nodepair_dists =\n{nodepair_dists}\n")

        expected_node_probs = [
            [0.5, 0.375],
            [0.75, 0.6875],
            [0.75, 0.6875],
            [0.5, 0.375],
        ]
        # sum for each node value and absolute difference for each node value, for each half-edge
        expected_edge_probs = [
            [1.25, 1.0625, 0.25, 0.3125],
            [1.25, 1.0625, 0.25, 0.3125],
            [1.5, 1.375, 0.0, 0.0],
            [1.5, 1.375, 0.0, 0.0],
            [1.25, 1.0625, 0.25, 0.3125],
            [1.25, 1.0625, 0.25, 0.3125],
        ]
        # sum for each node value and absolute difference for each node value, for each node pair
        expected_nodepair_probs = [
            [
                [1.0000, 0.7500, 0.0000, 0.0000],
                [1.2500, 1.0625, 0.2500, 0.3125],
                [1.2500, 1.0625, 0.2500, 0.3125],
                [1.0000, 0.7500, 0.0000, 0.0000],
            ],
            [
                [1.2500, 1.0625, 0.2500, 0.3125],
                [1.5000, 1.3750, 0.0000, 0.0000],
                [1.5000, 1.3750, 0.0000, 0.0000],
                [1.2500, 1.0625, 0.2500, 0.3125],
            ],
            [
                [1.2500, 1.0625, 0.2500, 0.3125],
                [1.5000, 1.3750, 0.0000, 0.0000],
                [1.5000, 1.3750, 0.0000, 0.0000],
                [1.2500, 1.0625, 0.2500, 0.3125],
            ],
            [
                [1.0000, 0.7500, 0.0000, 0.0000],
                [1.2500, 1.0625, 0.2500, 0.3125],
                [1.2500, 1.0625, 0.2500, 0.3125],
                [1.0000, 0.7500, 0.0000, 0.0000],
            ],
        ]
        self.assertEqual(node_probs.tolist(), expected_node_probs)
        self.assertEqual(edge_probs.tolist(), expected_edge_probs)
        self.assertEqual(nodepair_probs.tolist(), expected_nodepair_probs)

        expected_nodepair_dists = [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
        # Select half-edge node pairs
        expected_edge_dists = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        # Minimum of column, minimum of row, mean of column, mean of row,
        # stdev of column, stdev of row, for each node
        # stdev here uses n for normalization instead of n-1
        stdev_a = math.sqrt((1.5 * 1.5 + 0.5 * 0.5 + 0.5 * 0.5 + 1.5 * 1.5) / 4)
        stdev_b = math.sqrt((1.0 * 1.0 + 1.0 * 1.0) / 4)
        expected_node_dists = [
            [0.0, 0.0, 1.5, 1.5, stdev_a, stdev_a],
            [0.0, 0.0, 1.0, 1.0, stdev_b, stdev_b],
            [0.0, 0.0, 1.0, 1.0, stdev_b, stdev_b],
            [0.0, 0.0, 1.5, 1.5, stdev_a, stdev_a],
        ]
        np.testing.assert_array_almost_equal(node_dists.tolist(), expected_node_dists)
        self.assertEqual(edge_dists.tolist(), expected_edge_dists)
        self.assertEqual(nodepair_dists.tolist(), expected_nodepair_dists)


if __name__ == "__main__":
    ut.main()
