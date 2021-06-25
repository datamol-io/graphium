"""
Unit tests for the different layers of goli/dgl/dgl_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import numpy as np
import torch
import unittest as ut
import dgl
from copy import deepcopy

from goli.nn.dgl_layers import (
    GATLayer,
    GCNLayer,
    GINLayer,
    GatedGCNLayer,
    PNAConvolutionalLayer,
    PNAMessagePassingLayer,
    DGNConvolutionalLayer,
    DGNMessagePassingLayer,
)


class test_DGL_Layers(ut.TestCase):

    in_dim = 21
    out_dim = 11
    in_dim_edges = 13
    out_dim_edges = 17

    g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
    g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
    g1.ndata["h"] = torch.zeros(g1.num_nodes(), in_dim, dtype=float)
    g1.edata["e"] = torch.ones(g1.num_edges(), in_dim_edges, dtype=float)
    g2.ndata["h"] = torch.ones(g2.num_nodes(), in_dim, dtype=float)
    g2.edata["e"] = torch.zeros(g2.num_edges(), in_dim_edges, dtype=float)
    bg = dgl.batch([g1, g2])
    bg = dgl.add_self_loop(bg)
    bg.ndata["pos_dir"] = torch.randn_like(bg.ndata["h"])

    kwargs = {
        "activation": "relu",
        "dropout": 0.1,
        "normalization": "batch_norm",
    }

    def test_gcnlayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        layer = GCNLayer(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs).to(float)

        # Check the re-implementation of abstract methods
        self.assertFalse(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_ginlayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        layer = GINLayer(in_dim=self.in_dim, out_dim=self.out_dim, **self.kwargs).to(float)

        # Check the re-implementation of abstract methods
        self.assertFalse(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_gatlayer(self):

        num_heads = 3
        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        layer = GATLayer(in_dim=self.in_dim, out_dim=self.out_dim, num_heads=num_heads, **self.kwargs).to(
            float
        )

        # Check the re-implementation of abstract methods
        self.assertFalse(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, num_heads)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_gatedgcnlayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        e_in = bg.edata["e"]
        layer = GatedGCNLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            in_dim_edges=self.in_dim_edges,
            out_dim_edges=self.out_dim_edges,
            **self.kwargs,
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertTrue(layer.layer_outputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertIsInstance(layer.layer_outputs_edges, bool)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer without edges
        with self.assertRaises(TypeError):
            h = layer.forward(g=bg, h=h_in)

        # Apply layer with edges
        h, e = layer.forward(g=bg, h=h_in, e=e_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_pnaconvolutionallayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        e_in = bg.edata["e"]
        aggregators = ["mean", "max", "min", "lap", "std", "moment3", "moment4", "sum"]
        scalers = ["identity", "amplification", "attenuation"]

        layer = PNAConvolutionalLayer(
            in_dim=self.in_dim, out_dim=self.out_dim, aggregators=aggregators, scalers=scalers, **self.kwargs
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

        # Now try with edges
        layer = PNAConvolutionalLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            aggregators=aggregators,
            scalers=scalers,
            in_dim_edges=self.in_dim_edges,
            **self.kwargs,
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        with self.assertRaises(Exception):
            h = layer.forward(g=bg, h=h_in)

        h = layer.forward(g=bg, h=h_in, e=e_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_pnamessagepassinglayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        e_in = bg.edata["e"]
        aggregators = ["mean", "max", "min", "lap", "std", "moment3", "moment4", "sum"]
        scalers = ["identity", "amplification", "attenuation"]

        layer = PNAMessagePassingLayer(
            in_dim=self.in_dim, out_dim=self.out_dim, aggregators=aggregators, scalers=scalers, **self.kwargs
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

        # Now try with edges
        layer = PNAMessagePassingLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            aggregators=aggregators,
            scalers=scalers,
            in_dim_edges=self.in_dim_edges,
            **self.kwargs,
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        with self.assertRaises(Exception):
            h = layer.forward(g=bg, h=h_in)

        h = layer.forward(g=bg, h=h_in, e=e_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_dgnconvolutionallayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        e_in = bg.edata["e"]
        aggregators = [
            "mean",
            "max",
            "min",
            "lap",
            "std",
            "moment3",
            "moment4",
            "sum",
            "dir0/dx_abs",
            "dir1/dx_abs",
            "dir2/dx_no_abs",
            "dir1/smooth",
            "dir1/forward",
            "dir1/backward/0.5",
            "dir4/dx_abs/5",
        ]
        scalers = ["identity", "amplification", "attenuation"]

        layer = DGNConvolutionalLayer(
            in_dim=self.in_dim, out_dim=self.out_dim, aggregators=aggregators, scalers=scalers, **self.kwargs
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

        # Now try with edges
        layer = DGNConvolutionalLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            aggregators=aggregators,
            scalers=scalers,
            in_dim_edges=self.in_dim_edges,
            **self.kwargs,
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        with self.assertRaises(Exception):
            h = layer.forward(g=bg, h=h_in)

        h = layer.forward(g=bg, h=h_in, e=e_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

    def test_dgnmessagepassinglayer(self):

        bg = deepcopy(self.bg)
        h_in = bg.ndata["h"]
        e_in = bg.edata["e"]
        aggregators = [
            "mean",
            "max",
            "min",
            "lap",
            "std",
            "moment3",
            "moment4",
            "sum",
            "dir0/dx_abs",
            "dir1/dx_abs",
            "dir2/dx_no_abs",
            "dir1/smooth",
            "dir1/forward",
            "dir1/backward/0.5",
            "dir4/dx_abs/5",
        ]
        scalers = ["identity", "amplification", "attenuation"]

        layer = DGNMessagePassingLayer(
            in_dim=self.in_dim, out_dim=self.out_dim, aggregators=aggregators, scalers=scalers, **self.kwargs
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertFalse(layer.layer_inputs_edges)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        h = layer.forward(g=bg, h=h_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)

        # Now try with edges
        layer = DGNMessagePassingLayer(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            aggregators=aggregators,
            scalers=scalers,
            in_dim_edges=self.in_dim_edges,
            **self.kwargs,
        ).to(float)

        # Check the re-implementation of abstract methods
        self.assertTrue(layer.layer_supports_edges)
        self.assertTrue(layer.layer_inputs_edges)
        self.assertIsInstance(layer.layer_supports_edges, bool)
        self.assertIsInstance(layer.layer_inputs_edges, bool)
        self.assertFalse(layer.layer_outputs_edges)
        self.assertEqual(layer.out_dim_factor, 1)

        # Apply layer
        with self.assertRaises(Exception):
            h = layer.forward(g=bg, h=h_in)

        h = layer.forward(g=bg, h=h_in, e=e_in)
        self.assertEqual(h.shape[0], h_in.shape[0])
        self.assertEqual(h.shape[1], self.out_dim * layer.out_dim_factor)


if __name__ == "__main__":
    ut.main()
