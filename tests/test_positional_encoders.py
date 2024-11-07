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
Unit tests for the different datasets of graphium/features/featurizer.py
"""

import numpy as np
import unittest as ut
from rdkit import Chem
import datamol as dm
import torch
from torch_geometric.data import Data

import graphium
import graphium_cpp

from graphium.nn.encoders import laplace_pos_encoder, mlp_encoder, signnet_pos_encoder


# TODO: Test the MLP_encoder and signnet_pos_encoder


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
        for ii, mol in enumerate(self.smiles):
            adj = self.adjs[ii]
            for num_pos in [1, 2, 4]:  # Can't test too much eigs because of multiplicities
                for disconnected_comp in [True, False]:
                    err_msg = f"adj_id={ii}, num_pos={num_pos}, disconnected_comp={disconnected_comp}"

                    features = {
                        "laplacian_eigval": {
                            "pos_type": "laplacian_eigval",
                            "num_pos": num_pos,
                            "disconnected_comp": disconnected_comp,
                            "pos_level": "node",
                        },
                        "laplacian_eigvec": {
                            "pos_type": "laplacian_eigvec",
                            "num_pos": num_pos,
                            "disconnected_comp": disconnected_comp,
                            "pos_level": "node",
                        },
                    }
                    (
                        pos_encoding_names,
                        pos_encoding_tensor,
                    ) = graphium_cpp.positional_feature_options_to_tensor(features)

                    tensors = get_pe_tensors(mol, pos_encoding_tensor)
                    eigvals = tensors[4]
                    eigvecs = tensors[5]

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

                    if not ("." in mol):
                        print(
                            f"About to test eigvecs for smiles {mol}, num_pos {num_pos}, disconnected_comp {disconnected_comp}"
                        )
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
        for ii, mol in enumerate(self.smiles):
            adj = self.adjs[ii]
            for ksteps in [1, 2, 4]:
                err_msg = f"adj_id={ii}, ksteps={ksteps}"

                num_nodes = adj.shape[0]
                pos_kwargs = {"pos_type": "rw_return_probs", "ksteps": ksteps, "pos_level": "node"}
                features = {
                    "rw_return_probs": pos_kwargs,
                }
                (pos_encoding_names, pos_encoding_tensor) = graphium_cpp.positional_feature_options_to_tensor(
                    features
                )
                tensors = get_pe_tensors(mol, pos_encoding_tensor)
                rwse_embed = tensors[4]

                self.assertEqual(list(rwse_embed.shape), [num_nodes, ksteps], msg=err_msg)

    # TODO: work in progress

    """
    continue debugging here, see how to adapt the laplace_pos_encoder
    code running now, question is where to add the laplace_pos_encoder
    """

    def test_laplacian_eigvec_with_encoder(self):
        for ii, mol in enumerate(self.smiles):
            for num_pos in [2, 4, 8]:  # Can't test too much eigs because of multiplicities
                for disconnected_comp in [True, False]:
                    for model_type in ["Transformer", "DeepSet", "MLP"]:
                        err_msg = f"adj_id={ii}, num_pos={num_pos}, disconnected_comp={disconnected_comp}"

                        features = {
                            "laplacian_eigval": {
                                "pos_type": "laplacian_eigval",
                                "num_pos": num_pos,
                                "disconnected_comp": disconnected_comp,
                                "pos_level": "node",
                            },
                            "laplacian_eigvec": {
                                "pos_type": "laplacian_eigvec",
                                "num_pos": num_pos,
                                "disconnected_comp": disconnected_comp,
                                "pos_level": "node",
                            },
                        }
                        (
                            pos_encoding_names,
                            pos_encoding_tensor,
                        ) = graphium_cpp.positional_feature_options_to_tensor(features)

                        tensors = get_pe_tensors(mol, pos_encoding_tensor)

                        input_keys = ["laplacian_eigvec", "laplacian_eigval"]
                        in_dim = num_pos
                        hidden_dim = 64
                        out_dim = 64
                        num_layers = 1

                        num_nodes = tensors[2].size(0)
                        data_dict = {
                            # "feat": tensors[2],
                            # "edge_feat": tensors[3],
                            "laplacian_eigval": tensors[4].float(),
                            "laplacian_eigvec": tensors[5].float(),
                        }
                        # Create the PyG graph object `Data`
                        data = Data(
                            edge_index=tensors[0], edge_weight=tensors[1], num_nodes=num_nodes, **data_dict
                        )

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

                        hidden_embed = encoder(data, key_prefix=None)
                        assert "node" in hidden_embed.keys()
                        self.assertEqual(list(hidden_embed["node"].shape), [num_nodes, out_dim], msg=err_msg)


if __name__ == "__main__":
    ut.main()
