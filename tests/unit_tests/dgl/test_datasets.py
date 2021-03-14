"""
Unit tests for the different datasets of goli/dgl/datasets.py
"""
import os
import numpy as np
import pandas as pd
import torch
import unittest as ut
import dgl
from copy import deepcopy
from functools import partial

import goli
from goli.dgl.datasets import SmilesDataset
from goli.mol_utils.featurizer import mol_to_dglgraph

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(goli.__file__)))
MICRO_ZINC_PATH = os.path.join(BASE_PATH, "data/micro_ZINC/full.csv")
SMILES_COL = "SMILES"
LABELS_COLS = ["SA", "logp", "score"]


class test_SmilesDataset(ut.TestCase):
    def test_smilesdataset_basics(self):
        df = pd.read_csv(MICRO_ZINC_PATH)
        smiles = df[SMILES_COL]
        dtype = torch.float32
        other_dtype = torch.float64

        for ii in range(len(LABELS_COLS)):
            labels = df[LABELS_COLS[:ii]]
            for jj, weights in enumerate([None, df[LABELS_COLS[1]]**2]):
                for n_jobs in [0, 2, -1]:
                    err_msg = f'Error for ii={ii}, jj={jj}, n_jobs={n_jobs}'

                    dataset = SmilesDataset(
                        smiles=smiles,
                        labels=labels,
                        weights=weights,
                        smiles_transform=None,
                        collate_fn=None,
                        n_jobs=n_jobs,
                        dtype=dtype,
                    )

                    self.assertEqual(dataset.has_weights, weights is not None, msg=err_msg)
                    self.assertFalse(dataset.is_cuda, msg=err_msg)
                    self.assertIsNone(dataset.collate_fn, msg=err_msg)
                    self.assertEqual(dataset.dtype, dtype, msg=err_msg)
                    self.assertEqual(len(dataset), len(df), msg=err_msg)
                    self.assertEqual(dataset.to(dtype=other_dtype).dtype, other_dtype, msg=err_msg)

                    size = 100
                    choice = np.random.choice(len(dataset), size, replace=False)
                    if weights is None:
                        X, y = dataset[choice]
                    else:
                        X, y, w = dataset[choice]
                        self.assertEqual(len(w), size, msg=err_msg)
                    self.assertEqual(len(X), size, msg=err_msg)
                    self.assertListEqual(list(y.shape), [size, labels.shape[1]], msg=err_msg)
                    
    def test_smilesdataset_to_mol(self):
        df = pd.read_csv(MICRO_ZINC_PATH)
        smiles = df[SMILES_COL]
        dtype = torch.float32
        other_dtype = torch.float64
        to_mol = partial(mol_to_dglgraph, atom_property_list_float=['weight', 'valence'])

        for ii in range(len(LABELS_COLS)):
            labels = df[LABELS_COLS[:ii]]
            for jj, weights in enumerate([None, df[LABELS_COLS[1]]**2]):
                for n_jobs in [0, 2, -1]:
                    err_msg = f'Error for ii={ii}, jj={jj}, n_jobs={n_jobs}'

                    dataset = SmilesDataset(
                        smiles=smiles,
                        labels=labels,
                        weights=weights,
                        smiles_transform=to_mol,
                        collate_fn=None,
                        n_jobs=n_jobs,
                        dtype=dtype,
                    )

                    self.assertEqual(dataset.has_weights, weights is not None, msg=err_msg)
                    self.assertFalse(dataset.is_cuda, msg=err_msg)
                    self.assertIsNone(dataset.collate_fn, msg=err_msg)
                    self.assertEqual(dataset.dtype, dtype, msg=err_msg)
                    self.assertEqual(len(dataset), len(df), msg=err_msg)
                    self.assertEqual(dataset.to(dtype=other_dtype).dtype, other_dtype, msg=err_msg)

                    size = 100
                    choice = np.random.choice(len(dataset), size, replace=False)
                    if weights is None:
                        X, y = [dataset[ii][0] for ii in choice], [dataset[ii][1] for ii in choice]
                    else:
                        X, y, w = [dataset[ii][0] for ii in choice], [dataset[ii][1] for ii in choice], [dataset[ii][2] for ii in choice]
                        self.assertEqual(len(w), size, msg=err_msg)
                    self.assertEqual(len(X), size, msg=err_msg)
                    self.assertListEqual([len(y), len(y[0])], [size, labels.shape[1]], msg=err_msg)
                    

if __name__ == "__main__":
    ut.main()
