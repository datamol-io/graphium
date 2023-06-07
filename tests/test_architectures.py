"""
Unit tests for the different architectures of graphium/nn/architectures...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import torch
import unittest as ut
from copy import deepcopy
import sys
import traceback

from graphium.nn.architectures import FeedForwardNN, FeedForwardPyg, FullGraphMultiTaskNetwork
from graphium.nn.base_layers import FCLayer
from graphium.nn.residual_connections import (
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
)
from torch_geometric.data import Data, Batch

from graphium.utils.spaces import LAYERS_DICT


class test_FeedForwardNN(ut.TestCase):
    kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "normalization": "none",
        "dropout": 0.2,
        "name": "LNN",
        "layer_type": FCLayer,
    }

    norms = ["none", "batch_norm", "layer_norm"]

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

    def test_forward_norms(self):
        in_dim = 8
        out_dim = 16
        hidden_dims = [6, 6, 6, 6, 6]
        batch = 2

        for normalization in self.norms:
            this_kwargs = deepcopy(self.kwargs)
            this_kwargs["normalization"] = normalization
            err_msg = f"normalization = {normalization}"
            lnn = FeedForwardNN(
                in_dim=in_dim,
                out_dim=out_dim,
                hidden_dims=hidden_dims,
                residual_type="simple",
                residual_skip_steps=1,
                **this_kwargs,
            )

            self.assertEqual(len(lnn.layers), len(hidden_dims) + 1, msg=err_msg)
            self.assertEqual(lnn.layers[0].in_dim, in_dim, msg=err_msg)
            self.assertEqual(lnn.layers[1].in_dim, hidden_dims[0], msg=err_msg)
            self.assertEqual(lnn.layers[2].in_dim, hidden_dims[1], msg=err_msg)
            self.assertEqual(lnn.layers[3].in_dim, hidden_dims[2], msg=err_msg)

            feat = torch.FloatTensor(batch, in_dim)
            feat_out = lnn.forward(feat)

            self.assertListEqual(list(feat_out.shape), [batch, out_dim], msg=err_msg)

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])

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

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = lnn.forward(feat)

        self.assertListEqual(list(feat_out.shape), [batch, out_dim])


class test_FeedForwardGraph(ut.TestCase):
    kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "dropout": 0.2,
        "name": "LNN",
    }

    in_dim = 7
    out_dim = 11
    in_dim_edges = 13
    hidden_dims = [6, 6, 6, 6, 6]

    edge_idx1 = (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
    edge_idx2 = (torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0]))
    num_edges1 = len(edge_idx1[0])
    num_nodes1 = max(edge_idx1[0].max(), edge_idx1[1].max()) + 1
    num_edges2 = len(edge_idx2[0])
    num_nodes2 = max(edge_idx2[0].max(), edge_idx2[1].max()) + 1
    h1 = torch.zeros(num_nodes1, in_dim, dtype=torch.float32)
    e1 = torch.ones(num_edges1, in_dim_edges, dtype=torch.float32)
    h2 = torch.ones(num_nodes2, in_dim, dtype=torch.float32)
    e2 = torch.zeros(num_edges2, in_dim_edges, dtype=torch.float32)

    g1 = Data(feat=h1, edge_index=torch.stack(edge_idx1), edge_feat=e1)
    g2 = Data(feat=h2, edge_index=torch.stack(edge_idx2), edge_feat=e2)
    data_list = [g1, g2, deepcopy(g1), deepcopy(g2)]
    batch_pyg = Batch.from_data_list(data_list)
    num_nodes = batch_pyg.num_nodes
    num_edges = batch_pyg.num_edges
    batch_size = len(data_list)

    virtual_nodes = ["none", "mean", "sum"]
    norms = ["none", "batch_norm", "layer_norm"]
    pna_kwargs = {"aggregators": ["mean", "max", "sum"], "scalers": ["identity", "amplification"]}

    layers_kwargs = {
        "pyg:gin": {},
        "pyg:gine": {"in_dim_edges": in_dim_edges},
        "pyg:gated-gcn": {"in_dim_edges": in_dim_edges, "hidden_dims_edges": hidden_dims},
        "pyg:pna-msgpass#1": {"layer_kwargs": pna_kwargs, "in_dim_edges": 0},
        "pyg:pna-msgpass#2": {"layer_kwargs": pna_kwargs, "in_dim_edges": in_dim_edges},
    }

    def test_forward_no_residual(self):
        for residual_skip_steps in [1, 2]:
            for virtual_node in self.virtual_nodes:
                for normalization in self.norms:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}, normalization={normalization}"
                        layer_type = layer_name.split("#")[0]

                        # PYG
                        if layer_type.startswith("pyg:"):
                            layer_class = FeedForwardPyg
                            bg = deepcopy(self.batch_pyg)

                        gnn = layer_class(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="none",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            normalization=normalization,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        # gnn.to(torch.float32)

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

                        out_g = gnn.forward(bg)
                        feat_out = out_g["feat"]
                        edge_feat_out = out_g["edge_feat"]

                        out_dim_edges = (
                            gnn.full_dims_edges[-1] if gnn.full_dims_edges is not None else self.in_dim_edges
                        )

                        self.assertListEqual(
                            list(feat_out.shape), [self.num_nodes, self.out_dim], msg=err_msg
                        )
                        self.assertListEqual(
                            list(edge_feat_out.shape), [self.num_edges, out_dim_edges], msg=err_msg
                        )

    def test_forward_simple_residual(self):
        for residual_skip_steps in [1, 2]:
            for virtual_node in self.virtual_nodes:
                for normalization in self.norms:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}, normalization={normalization}"
                        layer_type = layer_name.split("#")[0]

                        # PYG
                        if layer_type.startswith("pyg:"):
                            layer_class = FeedForwardPyg
                            bg = deepcopy(self.batch_pyg)

                        gnn = layer_class(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="simple",
                            residual_skip_steps=1,
                            layer_type=layer_type,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        # gnn.to(torch.float32)

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

                        out_g = gnn.forward(bg)
                        feat_out = out_g["feat"]
                        edge_feat_out = out_g["edge_feat"]

                        out_dim_edges = (
                            gnn.full_dims_edges[-1] if gnn.full_dims_edges is not None else self.in_dim_edges
                        )

                        self.assertListEqual(
                            list(feat_out.shape), [self.num_nodes, self.out_dim], msg=err_msg
                        )
                        self.assertListEqual(
                            list(edge_feat_out.shape), [self.num_edges, out_dim_edges], msg=err_msg
                        )

    def test_forward_weighted_residual(self):
        for residual_skip_steps in [1, 2]:
            for virtual_node in self.virtual_nodes:
                for normalization in self.norms:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}, normalization={normalization}"
                        layer_type = layer_name.split("#")[0]

                        # PYG
                        if layer_type.startswith("pyg:"):
                            layer_class = FeedForwardPyg
                            bg = deepcopy(self.batch_pyg)

                        gnn = layer_class(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="weighted",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        # gnn.to(torch.float32)

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

                        out_g = gnn.forward(bg)
                        feat_out = out_g["feat"]
                        edge_feat_out = out_g["edge_feat"]

                        out_dim_edges = (
                            gnn.full_dims_edges[-1] if gnn.full_dims_edges is not None else self.in_dim_edges
                        )

                        self.assertListEqual(
                            list(feat_out.shape), [self.num_nodes, self.out_dim], msg=err_msg
                        )
                        self.assertListEqual(
                            list(edge_feat_out.shape), [self.num_edges, out_dim_edges], msg=err_msg
                        )

    def test_forward_concat_residual(self):
        for residual_skip_steps in [1, 2]:
            for virtual_node in self.virtual_nodes:
                for normalization in self.norms:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}, normalization={normalization}"
                        layer_type = layer_name.split("#")[0]

                        # PYG
                        if layer_type.startswith("pyg:"):
                            layer_class = FeedForwardPyg
                            bg = deepcopy(self.batch_pyg)

                        gnn = layer_class(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="concat",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        # gnn.to(torch.float32)

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

                        out_g = gnn.forward(bg)
                        feat_out = out_g["feat"]
                        edge_feat_out = out_g["edge_feat"]

                        out_dim_edges = (
                            gnn.full_dims_edges[-1] if gnn.full_dims_edges is not None else self.in_dim_edges
                        )

                        self.assertListEqual(
                            list(feat_out.shape), [self.num_nodes, self.out_dim], msg=err_msg
                        )
                        self.assertListEqual(
                            list(edge_feat_out.shape), [self.num_edges, out_dim_edges], msg=err_msg
                        )

    def test_forward_densenet_residual(self):
        for residual_skip_steps in [1, 2]:
            for virtual_node in self.virtual_nodes:
                for normalization in self.norms:
                    for layer_name, this_kwargs in self.layers_kwargs.items():
                        err_msg = f"virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}, normalization={normalization}"
                        layer_type = layer_name.split("#")[0]

                        # PYG
                        if layer_type.startswith("pyg:"):
                            layer_class = FeedForwardPyg
                            bg = deepcopy(self.batch_pyg)

                        gnn = layer_class(
                            in_dim=self.in_dim,
                            out_dim=self.out_dim,
                            hidden_dims=self.hidden_dims,
                            residual_type="densenet",
                            residual_skip_steps=residual_skip_steps,
                            layer_type=layer_type,
                            **this_kwargs,
                            **self.kwargs,
                        )
                        # gnn.to(torch.float32)

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

                        out_g = gnn.forward(bg)
                        feat_out = out_g["feat"]
                        edge_feat_out = out_g["edge_feat"]

                        out_dim_edges = (
                            gnn.full_dims_edges[-1] if gnn.full_dims_edges is not None else self.in_dim_edges
                        )

                        self.assertListEqual(
                            list(feat_out.shape), [self.num_nodes, self.out_dim], msg=err_msg
                        )
                        self.assertListEqual(
                            list(edge_feat_out.shape), [self.num_edges, out_dim_edges], msg=err_msg
                        )


if __name__ == "__main__":
    ut.main()
