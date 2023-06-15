"""
Unit tests for the different datasets of graphium/features/featurizer.py
"""

import numpy as np
import unittest as ut
from copy import deepcopy
from rdkit import Chem
import datamol as dm
import torch
from scipy.sparse import coo_matrix

from graphium.features.featurizer import GraphDict
from graphium.features.positional_encoding import graph_positional_encoder
from graphium.nn.encoders import laplace_pos_encoder, mlp_encoder, signnet_pos_encoder

# TODO: Test the MLP_encoder and signnet_pos_encoder


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

    def test_laplacian_eigvec_eigval(self):
        for ii, adj in enumerate(deepcopy(self.adjs)):
            for num_pos in [1, 2, 4]:  # Can't test too much eigs because of multiplicities
                for disconnected_comp in [True, False]:
                    err_msg = f"adj_id={ii}, num_pos={num_pos}, disconnected_comp={disconnected_comp}"

                    # returns a dictionary of computed pe
                    pos_kwargs = {
                        "pos_type": "laplacian_eigvec",
                        "num_pos": num_pos,
                        "disconnected_comp": disconnected_comp,
                        "pos_level": "node",
                    }
                    num_nodes = adj.shape[0]
                    eigvecs, cache = graph_positional_encoder(adj, num_nodes, pos_kwargs=pos_kwargs)
                    pos_kwargs["pos_type"] = "laplacian_eigval"
                    eigvals, cache = graph_positional_encoder(adj, num_nodes, pos_kwargs=pos_kwargs)

                    self.assertEqual(list(eigvecs.shape), [adj.shape[0], num_pos], msg=err_msg)
                    self.assertEqual(list(eigvals.shape), [adj.shape[0], num_pos], msg=err_msg)

                    # Compute eigvals and eigvecs
                    lap = np.diag(np.sum(adj, axis=1)) - adj
                    true_eigvals, true_eigvecs = np.linalg.eig(lap)
                    sort_idx = np.argsort(true_eigvals)
                    true_eigvals, true_eigvecs = true_eigvals[sort_idx], true_eigvecs[:, sort_idx]
                    true_eigvecs = true_eigvecs / (np.sum(true_eigvecs**2, axis=0, keepdims=True) + 1e-8)

                    true_num_pos = min(num_pos, len(true_eigvals))
                    true_eigvals, true_eigvecs = true_eigvals[:true_num_pos], true_eigvecs[:, :true_num_pos]

                    if not ("." in self.smiles[ii]):
                        np.testing.assert_array_almost_equal(
                            np.abs(true_eigvecs),
                            np.abs(eigvecs[:, :true_num_pos]),
                            decimal=6,
                            err_msg=err_msg,
                        )
                        self.assertAlmostEqual(np.sum(true_eigvecs[:, 1:]), 0, places=6, msg=err_msg)
                        np.testing.assert_array_almost_equal(
                            true_eigvals, eigvals[0, :true_num_pos], decimal=6, err_msg=err_msg
                        )

    # didn't actually check the exact computation result because the code was adapted
    def test_rwse(self):
        for ii, adj in enumerate(deepcopy(self.adjs)):
            for ksteps in [1, 2, 4]:
                err_msg = f"adj_id={ii}, ksteps={ksteps}"

                num_nodes = adj.shape[0]
                pos_kwargs = {"pos_type": "rw_return_probs", "ksteps": ksteps, "pos_level": "node"}
                rwse_embed, cache = graph_positional_encoder(adj, num_nodes, pos_kwargs=pos_kwargs)
                self.assertEqual(list(rwse_embed.shape), [num_nodes, ksteps], msg=err_msg)

    # TODO: work in progress

    """
    continue debugging here, see how to adapt the laplace_pos_encoder
    code running now, question is where to add the laplace_pos_encoder
    """

    def test_laplacian_eigvec_with_encoder(self):
        for ii, adj in enumerate(deepcopy(self.adjs)):
            for num_pos in [2, 4, 8]:  # Can't test too much eigs because of multiplicities
                for disconnected_comp in [True, False]:
                    for model_type in ["Transformer", "DeepSet", "MLP"]:
                        err_msg = f"adj_id={ii}, num_pos={num_pos}, disconnected_comp={disconnected_comp}"

                        # returns a dictionary of computed pe
                        pos_kwargs = {
                            "pos_type": "laplacian_eigvec",
                            "num_pos": num_pos,
                            "disconnected_comp": disconnected_comp,
                            "pos_level": "node",
                        }
                        num_nodes = adj.shape[0]
                        eigvecs, cache = graph_positional_encoder(adj, num_nodes, pos_kwargs=pos_kwargs)
                        pos_kwargs["pos_type"] = "laplacian_eigval"
                        eigvals, cache = graph_positional_encoder(adj, num_nodes, pos_kwargs=pos_kwargs)

                        input_keys = ["laplacian_eigvec", "laplacian_eigval"]
                        in_dim = num_pos
                        hidden_dim = 64
                        out_dim = 64
                        num_layers = 1

                        eigvecs = torch.from_numpy(eigvecs)
                        eigvals = torch.from_numpy(eigvals)

                        g = GraphDict(
                            {
                                "adj": coo_matrix(adj),
                                "data": {"laplacian_eigval": eigvals, "laplacian_eigvec": eigvecs},
                            }
                        )
                        batch = g.make_pyg_graph()

                        encoder = laplace_pos_encoder.LapPENodeEncoder(
                            input_keys=input_keys,
                            output_keys=["node"],
                            in_dim=in_dim,  # Size of Laplace PE embedding
                            hidden_dim=hidden_dim,
                            out_dim=out_dim,
                            model_type=model_type,  # 'Transformer' or 'DeepSet'
                            num_layers=num_layers,
                            num_layers_post=2,  # Num. layers to apply after pooling
                            dropout=0.1,
                            first_normalization=None,
                        )

                        hidden_embed = encoder(batch, key_prefix=None)
                        assert "node" in hidden_embed.keys()
                        self.assertEqual(list(hidden_embed["node"].shape), [num_nodes, out_dim], msg=err_msg)


if __name__ == "__main__":
    ut.main()
