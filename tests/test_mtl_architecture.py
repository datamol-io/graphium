"""
Unit tests for the different architectures of goli/nn/architectures...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import torch
import unittest as ut
from torch_geometric.data import Data, Batch
from copy import deepcopy
from itertools import product

import goli
from goli.config._loader import load_architecture
from goli.nn.architectures import TaskHeads, FullGraphMultiTaskNetwork
from goli.nn.base_layers import FCLayer

kwargs = {
    "activation": "relu",
    "last_activation": "none",
    "normalization": "none",
    "dropout": 0.2,
    "name": "LNN",
    "layer_type": FCLayer,
    "residual_type": "none",
    "residual_skip_steps": 1,
}

task_1_kwargs = {
    "out_dim": 5,
    "hidden_dims": [5, 6, 7],
}
task_2_kwargs = {
    "out_dim": 3,
    "hidden_dims": [8, 9, 10],
}
task_3_kwargs = {
    "out_dim": 4,
    "hidden_dims": [2, 2, 2],
}

task_1_params = {}
task_1_params.update(task_1_kwargs)
task_1_params.update(kwargs)
task_2_params = {}
task_2_params.update(task_2_kwargs)
task_2_params.update(kwargs)
task_3_params = {}
task_3_params.update(task_3_kwargs)
task_3_params.update(kwargs)


class test_TaskHeads(ut.TestCase):
    def test_task_heads_forward(self):
        in_dim = 4  # Dimension of the incoming data
        batch = 2

        task_heads_params = {"task_1": task_1_params, "task_2": task_2_params, "task_3": task_3_params}

        # Create the "multitask" network. Really it's just an input going to various FFNNs since there's nothing shared.
        multi_head_nn = TaskHeads(in_dim=in_dim, task_heads_kwargs=task_heads_params)

        # Test the sizes of the MLPs for each head
        # Head for task_1
        task_1_head = multi_head_nn.task_heads["task_1"]

        # Check the dimensions
        self.assertEqual(len(task_1_head.layers), len(task_1_kwargs["hidden_dims"]) + 1)
        self.assertEqual(task_1_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_1_head.layers[1].in_dim, task_1_kwargs["hidden_dims"][0])
        self.assertEqual(task_1_head.layers[2].in_dim, task_1_kwargs["hidden_dims"][1])
        self.assertEqual(task_1_head.layers[3].in_dim, task_1_kwargs["hidden_dims"][2])

        # Head for task_2
        task_2_head = multi_head_nn.task_heads["task_2"]

        # Check the dimensions
        self.assertEqual(len(task_2_head.layers), len(task_2_kwargs["hidden_dims"]) + 1)
        self.assertEqual(task_2_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_2_head.layers[1].in_dim, task_2_kwargs["hidden_dims"][0])
        self.assertEqual(task_2_head.layers[2].in_dim, task_2_kwargs["hidden_dims"][1])
        self.assertEqual(task_2_head.layers[3].in_dim, task_2_kwargs["hidden_dims"][2])

        # Head for task_3
        task_3_head = multi_head_nn.task_heads["task_3"]

        # Check the dimensions
        self.assertEqual(len(task_3_head.layers), len(task_3_kwargs["hidden_dims"]) + 1)
        self.assertEqual(task_3_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_3_head.layers[1].in_dim, task_3_kwargs["hidden_dims"][0])
        self.assertEqual(task_3_head.layers[2].in_dim, task_3_kwargs["hidden_dims"][1])
        self.assertEqual(task_3_head.layers[3].in_dim, task_3_kwargs["hidden_dims"][2])

        feat = torch.FloatTensor(batch, in_dim)
        feat_out = multi_head_nn.forward(feat)

        # Check the output: It's a per-task prediction!
        self.assertListEqual(list(feat_out["task_1"].shape), [batch, task_1_kwargs["out_dim"]])
        self.assertListEqual(list(feat_out["task_2"].shape), [batch, task_2_kwargs["out_dim"]])
        self.assertListEqual(list(feat_out["task_3"].shape), [batch, task_3_kwargs["out_dim"]])


class test_Multitask_NN(ut.TestCase):
    pyg_kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "normalization": "none",
        "dropout": 0.2,
        "name": "LNN",
    }

    in_dim = 7
    fullgraph_out_dim = 11
    in_dim_edges = 3
    hidden_dims = [6, 6, 6, 6, 6]

    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.randn(edge_idx1.max() + 1, in_dim, dtype=torch.float32)
    e1 = torch.randn(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.randn(edge_idx2.max() + 1, in_dim, dtype=torch.float32)
    e2 = torch.randn(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)
    # edge_idx1, e1 = add_self_loops(edge_idx1, e1)
    # edge_idx2, e2 = add_self_loops(edge_idx2, e2)
    g1 = Data(feat=x1, edge_index=edge_idx1, edge_feat=e1)
    g2 = Data(feat=x2, edge_index=edge_idx2, edge_feat=e2)
    bg = Batch.from_data_list([g1, g2])

    virtual_nodes = ["none", "mean", "sum"]
    norms = ["none", None, "batch_norm", "layer_norm"]

    gnn_layers_kwargs = {
        "pyg:gin": {},
        "pyg:gated-gcn": {"in_dim_edges": in_dim_edges, "hidden_dims_edges": hidden_dims},
    }

    def test_full_graph_multitask_forward(self):
        # Task heads
        task_heads_params = {"task_1": task_1_params, "task_2": task_2_params, "task_3": task_3_params}

        # Params for the network
        temp_dim_1 = 5
        temp_dim_2 = 7

        pre_nn_kwargs = dict(in_dim=self.in_dim, out_dim=temp_dim_1, hidden_dims=[4, 4, 4, 4, 4])

        post_nn_kwargs = dict(in_dim=temp_dim_2, out_dim=self.fullgraph_out_dim, hidden_dims=[3, 3, 3, 3])

        options = product(
            [["none"], ["sum"], ["mean", "max"]],
            [1, 2],
            self.virtual_nodes,
            self.norms,
            self.gnn_layers_kwargs.items(),
        )
        for pooling, residual_skip_steps, virtual_node, normalization, (layer_name, this_kwargs) in options:
            err_msg = f"pooling={pooling}, virtual_node={virtual_node}, layer_name={layer_name}, residual_skip_steps={residual_skip_steps}, normalization={normalization}"
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
                **self.pyg_kwargs,
            )

            multitask_graph_nn = FullGraphMultiTaskNetwork(
                task_heads_kwargs=task_heads_params,
                gnn_kwargs=gnn_kwargs,
                pre_nn_kwargs=pre_nn_kwargs,
                post_nn_kwargs=post_nn_kwargs,
            )

            bg = deepcopy(self.bg)
            feat_out = multitask_graph_nn.forward(bg)

            dim_1 = bg.num_nodes if pooling == ["none"] else bg.num_graphs
            # self.assertListEqual(list(feat_out.shape), [dim_1, self.out_dim], msg=err_msg)

            self.assertListEqual(
                list(feat_out["task_1"].shape),
                [dim_1, task_1_kwargs["out_dim"]],
                msg=err_msg,
            )
            self.assertListEqual(
                list(feat_out["task_2"].shape),
                [dim_1, task_2_kwargs["out_dim"]],
                msg=err_msg,
            )
            self.assertListEqual(
                list(feat_out["task_3"].shape),
                [dim_1, task_3_kwargs["out_dim"]],
                msg=err_msg,
            )


class test_FullGraphMultiTaskNetwork(ut.TestCase):
    in_dim_nodes = 7
    in_dim_edges = 13
    in_dims = {"feat": in_dim_nodes, "edge_feat": in_dim_edges}

    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.randn(edge_idx1.max() + 1, in_dim_nodes, dtype=torch.float32)
    e1 = torch.randn(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.randn(edge_idx2.max() + 1, in_dim_nodes, dtype=torch.float32)
    e2 = torch.randn(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)
    g1 = Data(feat=x1, edge_index=edge_idx1, edge_feat=e1)
    g2 = Data(feat=x2, edge_index=edge_idx2, edge_feat=e2)
    bg = Batch.from_data_list([g1, g2])

    def test_FullGraphMultiTaskNetwork_from_config(self):
        cfg = goli.load_config(name="zinc_default_multitask_pyg")

        # Initialize the network
        model_class, model_kwargs = load_architecture(cfg, in_dims=self.in_dims)

        multitask_full_graph_nn = model_class(**model_kwargs)

        # Test
        bg = deepcopy(self.bg)
        feat_out = multitask_full_graph_nn.forward(bg)

        dim_1 = self.bg.num_graphs

        self.assertListEqual(list(feat_out["task_1"].shape), [dim_1, task_1_kwargs["out_dim"]])
        self.assertListEqual(list(feat_out["task_2"].shape), [dim_1, task_2_kwargs["out_dim"]])
        self.assertListEqual(list(feat_out["task_3"].shape), [dim_1, task_3_kwargs["out_dim"]])


if __name__ == "__main__":
    ut.main()
