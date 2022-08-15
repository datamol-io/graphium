"""
Unit tests for the different layers of goli/nn/dgl_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy

from goli.nn.pyg_layers import (
    GINConvPyg,
    GINEConvPyg,
    GatedGCNPyg,
    PNAMessagePassingPyg,
    GPSLayerPyg,
)


class test_Pyg_Layers(ut.TestCase):

    in_dim = 21
    out_dim = 11
    in_dim_edges = 13
    out_dim_edges = 17

    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.zeros(edge_idx1.max() + 1, in_dim, dtype=torch.float32)
    e1 = torch.ones(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.ones(edge_idx2.max() + 1, in_dim, dtype=torch.float32)
    e2 = torch.zeros(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)
    # edge_idx1, e1 = add_self_loops(edge_idx1, e1)
    # edge_idx2, e2 = add_self_loops(edge_idx2, e2)
    g1 = Data(h=x1, edge_index=edge_idx1, edge_attr=e1)
    g2 = Data(h=x2, edge_index=edge_idx2, edge_attr=e2)
    bg = Batch.from_data_list([g1, g2])

    kwargs = {
        "activation": "relu",
        "dropout": 0.1,
        "normalization": "batch_norm",
    }

    def test_gpslayer(self):
        bg = deepcopy(self.bg)
        h_in = bg.h
        layer = GPSLayerPyg(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer. Should crash due to different dim for nodes vs edges
        with self.assertRaises(ValueError):
            bg = layer.forward(bg)

        # Create new edge attributes with same dim and check that it works
        bg.edge_attr = torch.zeros((bg.edge_attr.shape[0], self.in_dim), dtype=torch.float32)
        bg = layer.forward(bg)
        self.assertEqual(bg.h.shape[0], h_in.shape[0])
        self.assertEqual(bg.h.shape[1], self.out_dim * layer.out_dim_factor)


    def test_ginlayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.h
        layer = GINConvPyg(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs)

        # Check the re-implementation of abstract methods
        self.assertFalse(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        bg = layer.forward(bg)
        self.assertEqual(bg.h.shape[0], h_in.shape[0])
        self.assertEqual(bg.h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_ginelayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.h
        layer = GINEConvPyg(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer. Should crash due to different dim for nodes vs edges
        with self.assertRaises(ValueError):
            bg = layer.forward(bg)

        # Create new edge attributes with same dim and check that it works
        bg.edge_attr = torch.zeros((bg.edge_attr.shape[0], self.in_dim), dtype=torch.float32)
        bg = layer.forward(bg)
        self.assertEqual(bg.h.shape[0], h_in.shape[0])
        self.assertEqual(bg.h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_gatedgcnlayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.h
        e_in = bg.edge_attr
        layer = GatedGCNPyg(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            in_dim_edges=self.in_dim_edges,
            out_dim_edges=self.out_dim_edges,
            **self.kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertTrue(layer.layer_outputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertIsInstance(layer.layer_outputs_edges, bool)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer with edges
        bg2 = layer.forward(bg)
        self.assertEqual(bg2.h.shape[0], h_in.shape[0])
        self.assertEqual(bg2.h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_pnamessagepassinglayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.h
        aggregators = ["mean", "max", "min", "std", "sum"]
        scalers = ["identity", "amplification", "attenuation"]

        layer = PNAMessagePassingPyg(
            in_dim=self.in_dim, out_dim=self.out_dim, aggregators=aggregators, scalers=scalers, **self.kwargs
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        # self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        bg2 = layer.forward(bg)
        self.assertEqual(bg2.h.shape[0], h_in.shape[0])
        self.assertEqual(bg2.h.shape[1], self.out_dim * layer.out_dim_factor)

        # Now try with edges
        bg = deepcopy(self.bg)
        layer = PNAMessagePassingPyg(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            aggregators=aggregators,
            scalers=scalers,
            in_dim_edges=self.in_dim_edges,
            **self.kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        # self.assertTrue(layer.layer_inputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        # self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        bg2 = layer.forward(bg)
        self.assertEqual(bg2.h.shape[0], h_in.shape[0])
        self.assertEqual(bg2.h.shape[1], self.out_dim * layer.out_dim_factor)
        self.assertTrue((bg2.edge_attr == self.bg.edge_attr).all)


if __name__ == "__main__":
    ut.main()
