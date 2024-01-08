"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


"""
Unit tests for the implementation of canonical reordering of molecules in `dataset.py` and `datamodule.py`
"""

import unittest as ut
from copy import deepcopy
import torch.nn as nn
import torch

from torch_geometric.data import Batch, Data

from graphium.utils.enums import TaskLevel as TL
from graphium.data.multilevel_utils import get_canonical_ranks_pair


class test_task_level(ut.TestCase):
    def test_task_level_enum(self):
        # Test string conversion
        self.assertEqual(str(TL.GRAPH), "graph")
        self.assertEqual(str(TL.NODE), "node")
        self.assertEqual(str(TL.EDGE), "edge")
        self.assertEqual(str(TL.NODEPAIR), "nodepair")

        # Test from_str conversion with str
        self.assertEqual(TL.from_str("graph"), TL.GRAPH)
        self.assertEqual(TL.from_str("node"), TL.NODE)
        self.assertEqual(TL.from_str("edge"), TL.EDGE)
        self.assertEqual(TL.from_str("nodepair"), TL.NODEPAIR)

        # Test from_str conversion with None
        self.assertEqual(TL.from_str(None), TL.GRAPH)

        # Test from_str conversion with TL
        self.assertEqual(TL.from_str(TL.GRAPH), TL.GRAPH)
        self.assertEqual(TL.from_str(TL.NODE), TL.NODE)
        self.assertEqual(TL.from_str(TL.EDGE), TL.EDGE)
        self.assertEqual(TL.from_str(TL.NODEPAIR), TL.NODEPAIR)


class test_node_reordering(ut.TestCase):
    def test_get_canonical_ranks_pair(self):
        all_canonical_ranks = [[1, 2, 3], [2, 3, 1], [3, 2, 1], [1, 2, 3], None, [2, 3, 1]]
        all_task_levels = [TL.NODE, TL.NODE, TL.EDGE, TL.NODEPAIR, TL.NODE, TL.GRAPH]

        # Check when they all map to the 1st element
        unique_ids_inv = [0, 0, 0, 0, 0, 0]
        canonical_ranks_pair = get_canonical_ranks_pair(all_canonical_ranks, all_task_levels, unique_ids_inv)
        self.assertListEqual(
            canonical_ranks_pair,
            [
                None,
                ([1, 2, 3], [2, 3, 1]),
                ([1, 2, 3], [3, 2, 1]),
                None,
                None,
                None,
            ],
        )

        # Check when they all map to the 2nd element
        unique_ids_inv = [1, 1, 1, 1, 1, 1]
        canonical_ranks_pair = get_canonical_ranks_pair(all_canonical_ranks, all_task_levels, unique_ids_inv)
        self.assertListEqual(
            canonical_ranks_pair,
            [
                ([2, 3, 1], [1, 2, 3]),
                None,
                ([2, 3, 1], [3, 2, 1]),
                ([2, 3, 1], [1, 2, 3]),
                None,
                None,
            ],
        )

        # Check when they all map to the each other
        unique_ids_inv = [0, 1, 2, 3, 4, 5]
        canonical_ranks_pair = get_canonical_ranks_pair(all_canonical_ranks, all_task_levels, unique_ids_inv)
        self.assertListEqual(
            canonical_ranks_pair,
            [
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        )


if __name__ == "__main__":
    ut.main()
