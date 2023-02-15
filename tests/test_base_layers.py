"""
Unit tests for the different layers of goli/nn/base_layers
"""

from ast import Assert
import numpy as np
import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy

from goli.nn.base_layers import DropPath


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

    g1 = Data(h=x1, edge_index=edge_idx1, edge_attr=e1)
    g2 = Data(h=x2, edge_index=edge_idx2, edge_attr=e2)
    bg = Batch.from_data_list([g1, g2])

    def test_droppath_layer(self):
        bg = deepcopy(self.bg)
        h_in = bg.h
        layer = DropPath(drop_rate=0.5)
        h_out = layer.forward(h_in, bg.batch, on_ipu=False)
        self.assertEqual(h_out.shape, h_in.shape)
