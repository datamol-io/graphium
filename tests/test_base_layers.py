"""
Unit tests for the different layers of graphium/nn/base_layers
"""

import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy

from graphium.nn.base_layers import DropPath, TransformerEncoderLayerMup
from graphium.ipu.to_dense_batch import to_dense_batch, to_sparse_batch


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

        feat_dense, key_padding_mask, idx = to_dense_batch(
            feat_in,
            batch=bg.batch,
            batch_size=self.batch_size,
            max_num_nodes_per_graph=self.max_num_nodes_per_graph,
            drop_nodes_last_graph=False,
        )
        attn_mask = None
        key_padding_mask = ~key_padding_mask

        h_out_dense = layer.forward(feat_dense)

        h_out = to_sparse_batch(h_out_dense, mask_idx=idx)

        self.assertEqual(h_out.shape, feat_in.shape)
