"""
Unit tests for the different layers of goli/dgl/dgl_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import torch
import unittest as ut
import dgl
from copy import deepcopy

from goli.nn.architectures import FeedForwardNN, FeedForwardDGL, FullDGLNetwork, FullDGLSiameseNetwork
from goli.nn.base_layers import FCLayer
from goli.nn.residual_connections import (
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
)


class test_FeedForwardNN(ut.TestCase):

    kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "batch_norm": False,
        "dropout": 0.2,
        "name": "LNN",
        "layer_type": FCLayer,
    }

    def test_forward_no_residual(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [5, 6, 7]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="none",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_simple_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="simple",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_simple_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="simple",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_concat_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="concat",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 2 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 2 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 2 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 2 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_concat_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="concat",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, 1 * hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 1 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 2 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 1 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 2 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_densenet_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="densenet",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 2 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 3 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 4 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 5 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_densenet_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="densenet",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, 1 * hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, 2 * hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, 1 * hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, 3 * hidden_dims[4])

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_weighted_residual_1(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="weighted",
            residual_skip_steps=1,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, hidden_dims[4])

        self.assertEqual(len(lnn.residual_layer.residual_list), len(hidden_dims))

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_forward_weighted_residual_2(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        lnn = FeedForwardNN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="weighted",
            residual_skip_steps=2,
            **self.kwargs,
        )

        self.assertEqual(len(lnn.layers), len(hidden_dims) + 1)
        self.assertEqual(lnn.layers[0].in_dim, in_dim)
        self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2])
        self.assertEqual(lnn.layers[4].in_dim, hidden_dims[3])
        self.assertEqual(lnn.layers[5].in_dim, hidden_dims[4])

        self.assertEqual(len(lnn.residual_layer.residual_list), (len(hidden_dims) // 2 + 1))

        h = torch.FloatTensor(batch, in_dim)
        h_out = lnn.forward(h)

        self.assertListEqual(list(h_out.shape), [batch, out_dim])


class test_FeedForwardDGL(ut.TestCase):

    kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "batch_norm": False,
        "dropout": 0.2,
        "name": "LNN",
    }

    in_dim = 7
    out_dim = 11
    in_dim_edges = 13
    hidden_dims = [6, 6, 6, 6, 6]

    g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
    g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
    g1.ndata["h"] = torch.zeros(g1.num_nodes(), in_dim, dtype=torch.float32)
    g1.edata["e"] = torch.ones(g1.num_edges(), in_dim_edges, dtype=torch.float32)
    g2.ndata["h"] = torch.ones(g2.num_nodes(), in_dim, dtype=torch.float32)
    g2.edata["e"] = torch.zeros(g2.num_edges(), in_dim_edges, dtype=torch.float32)
    bg = dgl.batch([g1, g2])
    bg = dgl.add_self_loop(bg)

    virtual_nodes = ["none", "mean", "sum"]
    pna_kwargs = {"aggregators": ["mean", "max", "sum"], "scalers": ["identity", "amplification"]}

    layers_kwargs = {
        "gcn": {},
        "gin": {},
        "gat": {"num_heads": 3},
        "gated-gcn": {"in_dim_edges": in_dim_edges, "hidden_dims_edges": hidden_dims},
        "pna-conv": {**pna_kwargs},
        "pna-msgpass#1": {**pna_kwargs, "in_dim_edges": 0},
        "pna-msgpass#2": {**pna_kwargs, "in_dim_edges": in_dim_edges},
    }

    def test_dgl_forward_no_residual(self):

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}"
                        layer_type = layer_name.split("#")[0]

                        gnn = FeedForwardDGL(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="none",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            pooling=pooling,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        gnn.to(torch.float32)

                        self.assertIsInstance(gnn.residual_layer, ResidualConnectionNone)
                        self.assertEqual(len(gnn.layers), len(self.hidden_dims) + 1, msg=err_msg)
                        self.assertEqual(gnn.layers[0].out_dim, self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[1].out_dim, self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[2].out_dim, self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[3].out_dim, self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[4].out_dim, self.hidden_dims[4], msg=err_msg)
                        self.assertEqual(gnn.layers[5].out_dim, self.out_dim, msg=err_msg)

                        f = gnn.layers[0].out_dim_factor
                        self.assertEqual(gnn.layers[0].in_dim, self.in_dim, msg=err_msg)
                        self.assertEqual(gnn.layers[1].in_dim, f * self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[2].in_dim, f * self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[3].in_dim, f * self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[4].in_dim, f * self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[5].in_dim, f * self.hidden_dims[4], msg=err_msg)

                        bg = deepcopy(self.bg)
                        h_out = gnn.forward(bg)

                        dim_1 = bg.num_nodes() if pooling == ["none"] else 1
                        self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)

    def test_dgl_forward_simple_residual(self):

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}"
                        layer_type = layer_name.split("#")[0]

                        gnn = FeedForwardDGL(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="simple",
                            residual_skip_steps=1,
                            layer_type=layer_type,
                            pooling=pooling,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        gnn.to(torch.float32)

                        self.assertIsInstance(gnn.residual_layer, ResidualConnectionSimple)
                        self.assertEqual(len(gnn.layers), len(self.hidden_dims) + 1, msg=err_msg)
                        self.assertEqual(gnn.layers[0].out_dim, self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[1].out_dim, self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[2].out_dim, self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[3].out_dim, self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[4].out_dim, self.hidden_dims[4], msg=err_msg)
                        self.assertEqual(gnn.layers[5].out_dim, self.out_dim, msg=err_msg)

                        f = gnn.layers[0].out_dim_factor
                        self.assertEqual(gnn.layers[0].in_dim, self.in_dim, msg=err_msg)
                        self.assertEqual(gnn.layers[1].in_dim, f * self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[2].in_dim, f * self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[3].in_dim, f * self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[4].in_dim, f * self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[5].in_dim, f * self.hidden_dims[4], msg=err_msg)

                        bg = deepcopy(self.bg)
                        h_out = gnn.forward(bg)

                        dim_1 = bg.num_nodes() if pooling == ["none"] else 1
                        self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)

    def test_dgl_forward_weighted_residual(self):

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}"
                        layer_type = layer_name.split("#")[0]

                        gnn = FeedForwardDGL(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="weighted",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            pooling=pooling,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        gnn.to(torch.float32)

                        self.assertIsInstance(gnn.residual_layer, ResidualConnectionWeighted)
                        self.assertEqual(len(gnn.layers), len(self.hidden_dims) + 1, msg=err_msg)
                        self.assertEqual(gnn.layers[0].out_dim, self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[1].out_dim, self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[2].out_dim, self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[3].out_dim, self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[4].out_dim, self.hidden_dims[4], msg=err_msg)
                        self.assertEqual(gnn.layers[5].out_dim, self.out_dim, msg=err_msg)

                        f = gnn.layers[0].out_dim_factor
                        self.assertEqual(gnn.layers[0].in_dim, self.in_dim, msg=err_msg)
                        self.assertEqual(gnn.layers[1].in_dim, f * self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[2].in_dim, f * self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[3].in_dim, f * self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[4].in_dim, f * self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[5].in_dim, f * self.hidden_dims[4], msg=err_msg)

                        bg = deepcopy(self.bg)
                        h_out = gnn.forward(bg)

                        dim_1 = bg.num_nodes() if pooling == ["none"] else 1
                        self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)

    def test_dgl_forward_concat_residual(self):

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}"
                        layer_type = layer_name.split("#")[0]

                        gnn = FeedForwardDGL(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="concat",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            pooling=pooling,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        gnn.to(torch.float32)

                        self.assertIsInstance(gnn.residual_layer, ResidualConnectionConcat)
                        self.assertEqual(len(gnn.layers), len(self.hidden_dims) + 1, msg=err_msg)
                        self.assertEqual(gnn.layers[0].out_dim, self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[1].out_dim, self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[2].out_dim, self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[3].out_dim, self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[4].out_dim, self.hidden_dims[4], msg=err_msg)
                        self.assertEqual(gnn.layers[5].out_dim, self.out_dim, msg=err_msg)

                        f = gnn.layers[0].out_dim_factor
                        f2 = [2 * f if ((ii % residual_skip_steps) == 0 and ii > 0) else f for ii in range(6)]
                        self.assertEqual(gnn.layers[0].in_dim, self.in_dim, msg=err_msg)
                        self.assertEqual(gnn.layers[1].in_dim, f2[0] * self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[2].in_dim, f2[1] * self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[3].in_dim, f2[2] * self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[4].in_dim, f2[3] * self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[5].in_dim, f2[4] * self.hidden_dims[4], msg=err_msg)

                        bg = deepcopy(self.bg)
                        h_out = gnn.forward(bg)

                        dim_1 = bg.num_nodes() if pooling == ["none"] else 1
                        self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)

    def test_dgl_forward_densenet_residual(self):

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}"
                        layer_type = layer_name.split("#")[0]

                        gnn = FeedForwardDGL(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="densenet",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            pooling=pooling,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        gnn.to(torch.float32)

                        self.assertIsInstance(gnn.residual_layer, ResidualConnectionDenseNet)
                        self.assertEqual(len(gnn.layers), len(self.hidden_dims) + 1, msg=err_msg)
                        self.assertEqual(gnn.layers[0].out_dim, self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[1].out_dim, self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[2].out_dim, self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[3].out_dim, self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[4].out_dim, self.hidden_dims[4], msg=err_msg)
                        self.assertEqual(gnn.layers[5].out_dim, self.out_dim, msg=err_msg)

                        f = gnn.layers[0].out_dim_factor
                        f2 = [
                            ((ii // residual_skip_steps) + 1) * f
                            if ((ii % residual_skip_steps) == 0 and ii > 0)
                            else f
                            for ii in range(6)
                        ]
                        self.assertEqual(gnn.layers[0].in_dim, self.in_dim, msg=err_msg)
                        self.assertEqual(gnn.layers[1].in_dim, f2[0] * self.hidden_dims[0], msg=err_msg)
                        self.assertEqual(gnn.layers[2].in_dim, f2[1] * self.hidden_dims[1], msg=err_msg)
                        self.assertEqual(gnn.layers[3].in_dim, f2[2] * self.hidden_dims[2], msg=err_msg)
                        self.assertEqual(gnn.layers[4].in_dim, f2[3] * self.hidden_dims[3], msg=err_msg)
                        self.assertEqual(gnn.layers[5].in_dim, f2[4] * self.hidden_dims[4], msg=err_msg)

                        bg = deepcopy(self.bg)
                        h_out = gnn.forward(bg)

                        dim_1 = bg.num_nodes() if pooling == ["none"] else 1
                        self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)


class test_FullDGLNetwork(ut.TestCase):

    kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "batch_norm": False,
        "dropout": 0.2,
        "name": "LNN",
    }

    in_dim = 7
    out_dim = 11
    in_dim_edges = 13
    hidden_dims = [6, 6, 6, 6, 6]

    g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
    g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
    g1.ndata["h"] = torch.zeros(g1.num_nodes(), in_dim, dtype=torch.float32)
    g1.edata["e"] = torch.ones(g1.num_edges(), in_dim_edges, dtype=torch.float32)
    g2.ndata["h"] = torch.ones(g2.num_nodes(), in_dim, dtype=torch.float32)
    g2.edata["e"] = torch.zeros(g2.num_edges(), in_dim_edges, dtype=torch.float32)
    bg = dgl.batch([g1, g2])
    bg = dgl.add_self_loop(bg)

    virtual_nodes = ["none", "mean", "sum"]
    pna_kwargs = {"aggregators": ["mean", "max", "sum"], "scalers": ["identity", "amplification"]}

    gnn_layers_kwargs = {
        "gcn": {},
        "gin": {},
        "gat": {"num_heads": 3},
        "gated-gcn": {"in_dim_edges": in_dim_edges, "hidden_dims_edges": hidden_dims},
        "pna-conv": {**pna_kwargs},
        "pna-msgpass#1": {**pna_kwargs, "in_dim_edges": 0},
        "pna-msgpass#2": {**pna_kwargs, "in_dim_edges": in_dim_edges},
    }

    def test_full_network_densenet(self):

        temp_dim_1 = 5
        temp_dim_2 = 17

        pre_nn_kwargs = dict(in_dim=self.in_dim, out_dim=temp_dim_1, hidden_dims=[4, 4, 4, 4, 4])

        post_nn_kwargs = dict(in_dim=temp_dim_2, out_dim=self.out_dim, hidden_dims=[3, 3, 3, 3])

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for layer_name, this_kwargs in self.gnn_layers_kwargs.items():
                        err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}"
                        layer_type = layer_name.split("#")[0]

                        gnn_kwargs = dict(
                            in_dim=temp_dim_1,
                            out_dim=temp_dim_2,
                            hidden_dims=self.hidden_dims,
                            residual_type="densenet",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            pooling=pooling,
                            **this_kwargs,
                            **self.kwargs,
                        )

                        net = FullDGLNetwork(
                            pre_nn_kwargs=pre_nn_kwargs, gnn_kwargs=gnn_kwargs, post_nn_kwargs=post_nn_kwargs
                        )
                        bg = deepcopy(self.bg)
                        h_out = net.forward(bg)

                        dim_1 = bg.num_nodes() if pooling == ["none"] else 1
                        self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)


if __name__ == "__main__":
    ut.main()
