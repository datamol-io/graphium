"""
Unit tests for the different datasets of goli/features/featurizer.py
"""

import numpy as np
import unittest as ut
from copy import deepcopy
from rdkit import Chem
import datamol as dm

from goli.features.featurizer import (
    mol_to_adj_and_features,
    mol_to_dglgraph,
)
from goli.features.positional_encoding import graph_positional_encoder


class test_positional_encoder(ut.TestCase):

    smiles = [
        "C",
        "CC",
        "CC.CCCC",
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N",
        "O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5",
    ]
    mols = [dm.to_mol(s) for s in smiles]
    adjs = [Chem.rdmolops.GetAdjacencyMatrix(mol) for mol in mols]

    def test_laplacian_eigvec(self):

        for ii, adj in enumerate(deepcopy(self.adjs)):
            for num_pos in [1, 2, 4]: # Can't test too much eigs because of multiplicities
                for disconnected_comp in [True, False]:
                    err_msg = f"adj_id={ii}, num_pos={num_pos}, disconnected_comp={disconnected_comp}"
                    pos_enc = graph_positional_encoder(
                        adj, pos_type="laplacian_eigvec", num_pos=num_pos, disconnected_comp=disconnected_comp
                    )
                    self.assertEqual(list(pos_enc.shape), [adj.shape[0], num_pos], msg=err_msg)

                    # Compute eigvals and eigvecs
                    lap = np.diag(np.sum(adj, axis=1)) - adj
                    eigvals, eigvecs = np.linalg.eig(lap)
                    sort_idx = np.argsort(eigvals)
                    eigvals, eigvecs = eigvals[sort_idx], eigvecs[:, sort_idx]
                    eigvecs = eigvecs / (np.sum(eigvecs**2, axis=0, keepdims=True) + 1e-8)

                    true_num_pos = min(num_pos, len(eigvals))
                    eigvals, eigvecs = eigvals[:true_num_pos], eigvecs[:, :true_num_pos]
                    eigvecs = np.sign(eigvecs[0:1, :]) * eigvecs
                    pos_enc = (np.sign(pos_enc[0:1, :]) * pos_enc).numpy()

                    # Compare the positional encoding
                    if disconnected_comp and ("." in self.smiles[ii]):
                        self.assertGreater(np.max(np.abs(eigvecs - pos_enc)), 1e-3)
                    elif not ("." in self.smiles[ii]):
                        np.testing.assert_array_almost_equal(eigvecs, pos_enc[:, :true_num_pos], decimal=6, err_msg=err_msg)
                        

    def test_laplacian_eigvec_eigval(self):

        for ii, adj in enumerate(deepcopy(self.adjs)):
            for num_pos in [1, 2, 4]: # Can't test too much eigs because of multiplicities
                for disconnected_comp in [True, False]:
                    err_msg = f"adj_id={ii}, num_pos={num_pos}, disconnected_comp={disconnected_comp}"
                    pos_enc = graph_positional_encoder(
                        adj, pos_type="laplacian_eigvec_eigval", num_pos=num_pos, disconnected_comp=disconnected_comp
                    )
                    self.assertEqual(list(pos_enc.shape), [adj.shape[0], 2*num_pos], msg=err_msg)

                    # Compute eigvals and eigvecs
                    lap = np.diag(np.sum(adj, axis=1)) - adj
                    eigvals, eigvecs = np.linalg.eig(lap)
                    sort_idx = np.argsort(eigvals)
                    eigvals, eigvecs = eigvals[sort_idx], eigvecs[:, sort_idx]
                    eigvecs = eigvecs / (np.sum(eigvecs**2, axis=0, keepdims=True) + 1e-8)

                    true_num_pos = min(num_pos, len(eigvals))
                    eigvals, eigvecs = eigvals[:true_num_pos], eigvecs[:, :true_num_pos]
                    eigvecs = np.sign(eigvecs[0:1, :]) * eigvecs
                    pos_enc = (np.sign(pos_enc[0:1, :]) * pos_enc).numpy()

                    if not ("." in self.smiles[ii]):
                        np.testing.assert_array_almost_equal(eigvecs, pos_enc[:, :true_num_pos], decimal=6, err_msg=err_msg)
                        np.testing.assert_array_almost_equal(eigvals, pos_enc[0, num_pos:num_pos+true_num_pos], decimal=6, err_msg=err_msg)
                        



if __name__ == "__main__":
    ut.main()
