"""
Unit tests for the different datasets of goli/data/
"""
import pytest
import numpy as np
import pandas as pd
import unittest

import goli
from goli.data import SmilesDataset


class TestSmilesDataset(unittest.TestCase):

    smiles_col = "SMILES"
    label_cols = ["SA", "logp", "score"]

    def test_smilesdataset_basics(self):

        df: pd.DataFrame = goli.data.load_micro_zinc()
        smiles = df[self.smiles_col].to_list()

        for ii in range(len(self.label_cols)):
            labels = df[self.label_cols[:ii]].values

            for jj, weights in enumerate([None, df[self.label_cols[1]].values ** 2]):

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

                if dataset.weights is not None:
                    assert set(datum.keys()) == {"smiles", "features", "weights", "labels"}
                else:
                    assert set(datum.keys()) == {"smiles", "features", "labels"}

    # def test_smilesdataset_to_mol(self):
    #     df = pd.read_csv(MICRO_ZINC_PATH)
    #     smiles = df[SMILES_COL]
    #     dtype = torch.float32
    #     other_dtype = torch.float64
    #     to_mol = partial(mol_to_dglgraph, atom_property_list_float=["weight", "valence"])

    #     for ii in range(len(LABELS_COLS)):
    #         labels = df[LABELS_COLS[:ii]]
    #         for jj, weights in enumerate([None, df[LABELS_COLS[1]] ** 2]):
    #             for n_jobs in [0, 2, -1]:
    #                 err_msg = f"Error for ii={ii}, jj={jj}, n_jobs={n_jobs}"

    #                 dataset = SmilesDataset(
    #                     smiles=smiles,
    #                     labels=labels,
    #                     weights=weights,
    #                     smiles_transform=to_mol,
    #                     collate_fn=None,
    #                     n_jobs=n_jobs,
    #                     dtype=dtype,
    #                 )

    #                 self.assertEqual(dataset.has_weights, weights is not None, msg=err_msg)
    #                 self.assertFalse(dataset.is_cuda, msg=err_msg)
    #                 self.assertIsNone(dataset.collate_fn, msg=err_msg)
    #                 self.assertEqual(dataset.dtype, dtype, msg=err_msg)
    #                 self.assertEqual(len(dataset), len(df), msg=err_msg)
    #                 self.assertEqual(dataset.to(dtype=other_dtype).dtype, other_dtype, msg=err_msg)

    #                 size = 100
    #                 choice = np.random.choice(len(dataset), size, replace=False)
    #                 if weights is None:
    #                     X, y = [dataset[ii][0] for ii in choice], [dataset[ii][1] for ii in choice]
    #                 else:
    #                     X, y, w = (
    #                         [dataset[ii][0] for ii in choice],
    #                         [dataset[ii][1] for ii in choice],
    #                         [dataset[ii][2] for ii in choice],
    #                     )
    #                     self.assertEqual(len(w), size, msg=err_msg)
    #                 self.assertEqual(len(X), size, msg=err_msg)
    #                 self.assertListEqual([len(y), len(y[0])], [size, labels.shape[1]], msg=err_msg)

    # def test_DGLFromSmilesDataModule(self):
    #     df = pd.read_csv(MICRO_ZINC_PATH)
    #     smiles = df[SMILES_COL]
    #     dtype = torch.float32
    #     other_dtype = torch.float64
    #     to_mol = partial(mol_to_dglgraph, atom_property_list_float=["weight", "valence"])

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
    ut.main()
