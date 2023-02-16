"""
Unit tests for the different layers of goli/nn/dgl_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

from ast import Assert
import numpy as np
import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy

from goli.ipu.to_dense_batch import to_dense_batch
from goli.nn.pyg_layers import (
    GINConvPyg,
    GINEConvPyg,
    MPNNPlusPyg,
    GatedGCNPyg,
    PNAMessagePassingPyg,
    GPSLayerPyg,
    VirtualNodePyg,
)


class test_Pyg_Layers(ut.TestCase):
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

    def test_mpnnlayer(self):
        bg = deepcopy(self.bg)
        h_in = bg.h
        # need in_dim = out_dim for skip connection
        # mpnn layer accept different dimension for node and edge features
        layer = MPNNPlusPyg(
            in_dim=self.in_dim,
            out_dim=self.in_dim,
            use_edges=True,
            in_dim_edges=self.in_dim_edges,
            out_dim_edges=self.in_dim_edges,
            **self.kwargs,
        )

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertTrue(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Create new edge attributes with same dim and check that it works
        bg.edge_attr = torch.zeros((bg.edge_attr.shape[0], self.in_dim_edges), dtype=torch.float32)
        bg = layer.forward(bg)
        self.assertEqual(bg.h.shape[0], h_in.shape[0])
        self.assertEqual(bg.h.shape[1], self.in_dim * layer.out_dim_factor)

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

    def test_pooling_virtual_node(self):
        bg = deepcopy(self.bg)
        h_in = bg.h
        e_in = bg.edge_attr
        vn_h = 0
        edge_dim = bg.edge_attr.size()[1]

        vn_types = ["sum", "mean"]
        expected_vn_out = [(2, 10), (2, 10)]

        expected_h_out = [(), ()]
        expected_e_out = [(), ()]
        for vn_type, expected_shape, use_edges, global_latent in zip(vn_types, expected_vn_out, [False, True], [10, 10]):
            with self.subTest(vn_type=vn_type, expected_shape=expected_shape, use_edges=use_edges, global_latent=global_latent):
                if global_latent is not None:
                    vn_h = torch.zeros(global_latent)
                    vn_h = torch.tile(vn_h, (bg.num_graphs, 1))
                else:
                    vn_h = 0.0
                layer = VirtualNodePyg(
                    dim=self.in_dim,
                    global_latent=global_latent,
                    vn_type=vn_type,
                    use_edges=use_edges,
                    dim_edges=edge_dim,
                    **self.kwargs,
                )
                print(expected_shape, use_edges, global_latent, )
                h_out, vn_out, e_out = layer.forward(bg, h_in, vn_h, e=e_in)
                assert vn_out.shape == expected_shape
                if use_edges is False:
                    # i.e. that the node features have been updated
                    assert torch.equal(h_out, h_in) == False
                    # And that the edge features have not
                    assert torch.equal(e_out, e_in) == True
                else:
                    assert torch.equal(h_out, h_in) == False
                    assert torch.equal(e_out, e_in) == False


if __name__ == "__main__":
    ut.main()
