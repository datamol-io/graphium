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

from typing import Literal

import unittest as ut
from copy import deepcopy
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import datamol as dm

from graphium.utils.enums import TaskLevel as TL
from graphium.data.multilevel_utils import get_canonical_ranks_pair
from graphium.data.dataset import SingleTaskDataset


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

    def test_labels_reordering(self):

        labels = list(torch.randn(6, 3))
        all_canonical_ranks = [[1, 2, 3], [2, 3, 1], [3, 2, 1], [1, 2, 3], None, [2, 3, 1]]
        unique_ids_inv = [0, 0, 0, 0, 0, 0]
        task_level = TL.NODE
        all_task_levels = [task_level for _ in range(len(all_canonical_ranks))]
        canonical_ranks_pair = get_canonical_ranks_pair(all_canonical_ranks, all_task_levels, unique_ids_inv)
        reordered_labels = SingleTaskDataset.reorder_labels(labels, canonical_ranks_pair, task_level)

        self.assertListEqual(labels[0].tolist(), reordered_labels[0].tolist())
        self.assertListEqual(labels[1].tolist(), reordered_labels[1][[1, 2, 0]].tolist())
        self.assertListEqual(labels[2].tolist(), reordered_labels[2][[2, 1, 0]].tolist())
        self.assertListEqual(labels[3].tolist(), reordered_labels[3].tolist())
        self.assertListEqual(labels[4].tolist(), reordered_labels[4].tolist())
        self.assertListEqual(labels[5].tolist(), reordered_labels[5][[1, 2, 0]].tolist())

    def test_datamodule_reordering(self):

        # Make a dummy molecular dataset
        smiles = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC", "CCCCCCC"]
        labels = [np.random.randn(len(s)) for s in smiles]
        mols = [dm.to_mol(s) for s in smiles]
        ordered_smiles = [dm.to_smiles(m, canonical=False, with_atom_indices=True) for m in mols]

        # Reorder the molecules randomly
        np.random.seed(42)
        reordering_idx = [np.random.permutation(len(s)) for s in smiles]
        reordered_mols = [reorder_atoms_to_desired(m, idx.tolist()) for m, idx in zip(mols, reordering_idx)]
        reordered_smiles = [dm.to_smiles(m, canonical=False, with_atom_indices=True) for m in reordered_mols]


        pass


def ordered_carbon_smiles_generator(num_carbons, order: Literal["increasing", "random"]) -> str:
    """Generate a list of smiles with a specified number of carbons in the molecule.

    Args:
        num_carbons: The number of carbons in the molecule.
        order: The order of the carbons in the molecule. Either "increasing" or "random".

    Returns:
        smiles: A list of smiles with the specified number of carbons in the molecule.
    """
    if num_carbons == 1:
        return "[CH4:0]"
    elif num_carbons == 2:
        return "[CH3:0][CH3:1]"
    elif num_carbons > 2:

        if order == "increasing":
            return "".join([f"[CH3:{i}]" for i in range(num_carbons)])
        elif order == "random":
            idx = np.random.permutation(num_carbons)
            return "".join([f"[CH3:{idx[i]}]" for i in range(num_carbons)])
        else:
            raise ValueError(f"Unknown order: {order}")




from typing import List, Optional, Iterable
from rdkit.Chem import Mol, rdmolops
def reorder_atoms_to_desired(
    mol: Mol,
    desired_order: Iterable[int],
) -> Optional[Mol]:
    """Reorder the atoms in a mol. It ensures a single atom order for the same molecule,
    regardless of its original representation.

    Args:
        mol: a molecule.
        desired_order: The desired order of the atoms.

    Returns:
        mol: a molecule.
    """
    if mol.GetNumAtoms() == 0:
        return mol

    return rdmolops.RenumberAtoms(mol, list(desired_order))


if __name__ == "__main__":
    ut.main()
