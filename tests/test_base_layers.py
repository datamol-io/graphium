"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


"""
Unit tests for the different layers of graphium/nn/base_layers
"""

import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, dense_to_sparse
from copy import deepcopy

from graphium.nn.base_layers import DropPath, TransformerEncoderLayerMup


class test_Base_Layers(ut.TestCase):
    in_dim = 21
    out_dim = 11
    in_dim_edges = 13
    out_dim_edges = 17

    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.randn(edge_idx1.max() + 1, in_dim, dtype=torch.float32)
    e1 = torch.randn(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.randn(edge_idx2.max() + 1, in_dim, dtype=torch.float32)
    e2 = torch.randn(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)

    g1 = Data(feat=x1, edge_index=edge_idx1, edge_feat=e1)
    g2 = Data(feat=x2, edge_index=edge_idx2, edge_feat=e2)
    bg = Batch.from_data_list([g1, g2])

    batch_size = 2
    max_num_nodes_per_graph = max(x1.shape[0], x2.shape[0])

    # for drop_rate=0.5, test if the output shape is correct
    def test_droppath_layer_0p5(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        layer = DropPath(drop_rate=0.5)
        h_out = layer.forward(feat_in, bg.batch)
        self.assertEqual(h_out.shape, feat_in.shape)

    # for drop_rate=1.0, test if the output are all zeros
    def test_droppath_layer_1p0(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        zero_tesor = torch.zeros(feat_in.shape)
        layer = DropPath(drop_rate=1.0)
        h_out = layer.forward(feat_in, bg.batch)
        self.assertTrue(torch.allclose(zero_tesor, h_out.detach()))

    # for drop_rate=0.0, test if the output matches the original output
    def test_droppath_layer_0p0(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        layer = DropPath(drop_rate=0.0)
        h_out = layer.forward(feat_in, bg.batch)
        self.assertTrue(torch.allclose(feat_in.detach(), h_out.detach()))

    # test output shape is correct for TransformerEncoderLayerMup
    def test_transformer_encoder_layer_mup(self):
        bg = deepcopy(self.bg)
        feat_in = bg.feat
        layer = TransformerEncoderLayerMup(
            biased_attention=False, d_model=self.in_dim, nhead=1, dim_feedforward=4 * self.in_dim
        )

        feat_dense, key_padding_mask = to_dense_batch(
            feat_in,
            batch=bg.batch,
            batch_size=self.batch_size,
            max_num_nodes=self.max_num_nodes_per_graph,
        )

        key_padding_mask = ~key_padding_mask
        h_out_dense = layer.forward(feat_dense)
        h_out = h_out_dense[~key_padding_mask]

        self.assertEqual(h_out.shape, feat_in.shape)


if __name__ == "__main__":
    ut.main()
