"""
Unit tests for the different datasets of goli/data/
"""
import numpy as np
import pandas as pd
import functools

import unittest

import dgl

import goli
from goli.data import SmilesDataset
from goli.data import DGLFromSmilesDataModule
from goli.features.featurizer import mol_to_dglgraph

SMILES_COL = "SMILES"
LABELS_COLS = ["SA", "logp", "score"]


class TestSmilesDataset(unittest.TestCase):
    def test_smilesdataset_basics(self):

        df: pd.DataFrame = goli.data.load_micro_zinc()
        smiles = df[SMILES_COL].to_list()

        for ii in range(len(LABELS_COLS)):
            labels = df[LABELS_COLS[:ii]].values

            for jj, weights in enumerate([None, df[LABELS_COLS[1]].values ** 2]):

                err_msg = f"Error for ii={ii}, jj={jj}"

                dataset = SmilesDataset(
                    smiles=smiles,
                    labels=labels,
                    weights=weights,
                    smiles_transform=None,
                    collate_fn=None,
                )

                self.assertEqual(dataset.weights is not None, weights is not None, msg=err_msg)
                self.assertEqual(len(dataset), len(df), msg=err_msg)

                choice = np.random.randint(low=0, high=len(dataset))
                datum = dataset[choice]

                if weights is not None:
                    assert set(datum.keys()) == {"smiles", "features", "weights", "labels"}
                else:
                    assert set(datum.keys()) == {"smiles", "features", "labels"}

    def test_smilesdataset_to_mol(self):
        df: pd.DataFrame = goli.data.load_micro_zinc()
        smiles = df[SMILES_COL].to_list()

        transform_smiles = functools.partial(mol_to_dglgraph, atom_property_list_float=["weight", "valence"])

        for ii in range(len(LABELS_COLS)):
            labels = df[LABELS_COLS[:ii]].values

            for jj, weights in enumerate([None, df[LABELS_COLS[1]].values ** 2]):
                err_msg = f"Error for ii={ii}, jj={jj}"

                dataset = SmilesDataset(
                    smiles=smiles,
                    labels=labels,
                    weights=weights,
                    smiles_transform=transform_smiles,
                    collate_fn=None,
                )

                self.assertEqual(dataset.weights is not None, weights is not None, msg=err_msg)
                self.assertEqual(len(dataset), len(df), msg=err_msg)
                self.assertIsNone(dataset.collate_fn, msg=err_msg)

                choice = np.random.randint(low=0, high=len(dataset))
                datum = dataset[choice]
                print(datum)
                if weights is not None:
                    assert set(datum.keys()) == {"smiles", "features", "weights", "labels"}
                else:
                    assert set(datum.keys()) == {"smiles", "features", "labels"}
                assert isinstance(datum["features"], dgl.DGLGraph)

    # def test_DGLFromSmilesDataModule(self):
    #     df: pd.DataFrame = goli.data.load_micro_zinc()
    #     smiles = df[SMILES_COL].to_list()

    #     transform_smiles = functools.partial(mol_to_dglgraph, atom_property_list_float=["weight", "valence"])

    #     for ii in range(len(LABELS_COLS)):
    #         labels = df[LABELS_COLS[:ii]]
    #         for jj, weights in enumerate([None, df[LABELS_COLS[1]] ** 2]):
    #             for n_jobs in [0, 2, -1]:
    #                 err_msg = f"Error for ii={ii}, jj={jj}, n_jobs={n_jobs}"

    #                 dataset = DGLFromSmilesDataModule(
    #                     smiles=smiles,
    #                     labels=labels,
    #                     weights=weights,
    #                     n_jobs=n_jobs,
    #                     dtype=dtype,
    #                 )
    #                 dataset.setup()


if __name__ == "__main__":
    unittest.main()
