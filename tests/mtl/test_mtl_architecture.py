"""
Unit tests for the different architectures of goli/nn/architectures...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import torch
import unittest as ut
import dgl
from copy import deepcopy

import goli
from goli.config._loader import load_architecture
from goli.nn.architectures import TaskHeads, FullGraphMultiTaskNetwork, TaskHead, TaskHeadParams
from goli.nn.base_layers import FCLayer
from goli.nn.residual_connections import (
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
)

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
    "task_name": "task_1",
    "out_dim": 5,
    "hidden_dims": [5, 6, 7],
}
task_2_kwargs = {
    "task_name": "task_2",
    "out_dim": 3,
    "hidden_dims": [8, 9, 10],
}
task_3_kwargs = {
    "task_name": "task_3",
    "out_dim": 4,
    "hidden_dims": [2, 2, 2],
}

# The params to create the task head MLPs.
task_1_params = TaskHeadParams(
    **task_1_kwargs,
    **kwargs,
)
task_2_params = TaskHeadParams(
    **task_2_kwargs,
    **kwargs,
)
task_3_params = TaskHeadParams(
    **task_3_kwargs,
    **kwargs,
)

class test_TaskHeads(ut.TestCase):
    def test_task_head_forward(self):
        """Task heads are FeedForwardNNs, with the addition of a 'task_name' attribute.
        We want to make sure that they work as expected even though they are basically the same as FFNNs.

        This test matches `test_forward_no_residual` for the FeedForwardNN."""

        # Assume same setup as when testing the FFNN
        kwargs = {
            "activation": "relu",
            "last_activation": "none",
            "normalization": "none",
            "dropout": 0.2,
            "name": "LNN",
            "layer_type": FCLayer,
        }

        task_name = "task"
        in_dim = 8
        out_dim = 16
        hidden_dims = [5, 6, 7]
        batch = 2

        task_head = TaskHead(
            task_name=task_name,
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            residual_type="none",
            residual_skip_steps=1,
            **kwargs,
        )

        # Check that the task name is correct
        self.assertEqual(task_head.task_name, task_name)

        # Check that the layers were created as intended
        self.assertEqual(len(task_head.layers), len(hidden_dims) + 1)
        self.assertEqual(task_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_head.layers[1].in_dim, hidden_dims[0])
        self.assertEqual(task_head.layers[2].in_dim, hidden_dims[1])
        self.assertEqual(task_head.layers[3].in_dim, hidden_dims[2])

        h = torch.FloatTensor(batch, in_dim)
        h_out = task_head.forward(h)                # For now just inherits the forward() of FFNN

        # Check that the output has the correct dimensions
        self.assertListEqual(list(h_out.shape), [batch, out_dim])

    def test_task_heads_forward(self):

        in_dim = 4      # Dimension of the incoming data
        batch = 2

        task_heads_params = [task_1_params, task_2_params, task_3_params]

        # Create the "multitask" network. Really it's just an input going to various FFNNs since there's nothing shared.
        multi_head_nn = TaskHeads(in_dim=in_dim, task_heads_params=task_heads_params)

        # Test the sizes of the MLPs for each head
        # Head for task_1
        task_1_head = multi_head_nn.task_heads["task_1"]
        # Check that the task name is correct
        self.assertEqual(task_1_head.task_name, task_1_kwargs["task_name"])
        # Check the dimensions
        self.assertEqual(len(task_1_head.layers), len(task_1_kwargs["hidden_dims"]) + 1)
        self.assertEqual(task_1_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_1_head.layers[1].in_dim, task_1_kwargs["hidden_dims"][0])
        self.assertEqual(task_1_head.layers[2].in_dim, task_1_kwargs["hidden_dims"][1])
        self.assertEqual(task_1_head.layers[3].in_dim, task_1_kwargs["hidden_dims"][2])

        # Head for task_2
        task_2_head = multi_head_nn.task_heads["task_2"]
        # Check that the task name is correct
        self.assertEqual(task_2_head.task_name, task_2_kwargs["task_name"])
        # Check the dimensions
        self.assertEqual(len(task_2_head.layers), len(task_2_kwargs["hidden_dims"]) + 1)
        self.assertEqual(task_2_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_2_head.layers[1].in_dim, task_2_kwargs["hidden_dims"][0])
        self.assertEqual(task_2_head.layers[2].in_dim, task_2_kwargs["hidden_dims"][1])
        self.assertEqual(task_2_head.layers[3].in_dim, task_2_kwargs["hidden_dims"][2])

        # Head for task_3
        task_3_head = multi_head_nn.task_heads["task_3"]
        # Check that the task name is correct
        self.assertEqual(task_3_head.task_name, task_3_kwargs["task_name"])
        # Check the dimensions
        self.assertEqual(len(task_3_head.layers), len(task_3_kwargs["hidden_dims"]) + 1)
        self.assertEqual(task_3_head.layers[0].in_dim, in_dim)
        self.assertEqual(task_3_head.layers[1].in_dim, task_3_kwargs["hidden_dims"][0])
        self.assertEqual(task_3_head.layers[2].in_dim, task_3_kwargs["hidden_dims"][1])
        self.assertEqual(task_3_head.layers[3].in_dim, task_3_kwargs["hidden_dims"][2])

        h = torch.FloatTensor(batch, in_dim)
        h_out = multi_head_nn.forward(h)

        # Check the output: It's a per-task prediction!
        self.assertListEqual(list(h_out[task_1_kwargs["task_name"]].shape), [batch, task_1_kwargs["out_dim"]])
        self.assertListEqual(list(h_out[task_2_kwargs["task_name"]].shape), [batch, task_2_kwargs["out_dim"]])
        self.assertListEqual(list(h_out[task_3_kwargs["task_name"]].shape), [batch, task_3_kwargs["out_dim"]])


class test_Multitask_NN(ut.TestCase):
    fulldgl_kwargs = {
        "activation": "relu",
        "last_activation": "none",
        "normalization": "none",
        "dropout": 0.2,
        "name": "LNN",
    }

    in_dim = 7
    fulldgl_out_dim = 11
    in_dim_edges = 13
    hidden_dims = [6, 6, 6, 6, 6]

    g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
    g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
    g1.ndata["feat"] = torch.zeros(g1.num_nodes(), in_dim, dtype=torch.float32)
    g1.edata["feat"] = torch.ones(g1.num_edges(), in_dim_edges, dtype=torch.float32)
    g2.ndata["feat"] = torch.ones(g2.num_nodes(), in_dim, dtype=torch.float32)
    g2.edata["feat"] = torch.zeros(g2.num_edges(), in_dim_edges, dtype=torch.float32)
    batch = [g1, g2, deepcopy(g1), deepcopy(g2)]
    batch = [dgl.add_self_loop(g) for g in batch]
    bg = dgl.batch(batch)

    virtual_nodes = ["none", "mean", "sum"]
    norms = ["none", None, "batch_norm", "layer_norm"]
    pna_kwargs = {"aggregators": ["mean", "max", "sum"], "scalers": ["identity", "amplification"]}

    gnn_layers_kwargs = {
        "gcn": {},
        "gin": {},
        "gat": {"layer_kwargs": {"num_heads": 3}},
        "gated-gcn": {"in_dim_edges": in_dim_edges, "hidden_dims_edges": hidden_dims},
        "pna-conv": {"layer_kwargs": pna_kwargs},
        "pna-msgpass#1": {"layer_kwargs": pna_kwargs, "in_dim_edges": 0},
        "pna-msgpass#2": {"layer_kwargs": pna_kwargs, "in_dim_edges": in_dim_edges},
    }

    def test_fulldgl_multitask_forward(self):
        # Task heads
        task_heads_params = [task_1_params, task_2_params, task_3_params]

        # Params for the FullDGLNetwork
        temp_dim_1 = 5
        temp_dim_2 = 17

        pre_nn_kwargs = dict(in_dim=self.in_dim, out_dim=temp_dim_1, hidden_dims=[4, 4, 4, 4, 4])

        post_nn_kwargs = dict(in_dim=temp_dim_2, out_dim=self.fulldgl_out_dim, hidden_dims=[3, 3, 3, 3])

        for pooling in [["none"], ["sum"], ["mean", "s2s", "max"]]:
            for residual_skip_steps in [1, 2, 3]:
                for virtual_node in self.virtual_nodes:
                    for normalization in self.norms:
                        for layer_name, this_kwargs in self.gnn_layers_kwargs.items():
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
                                **self.fulldgl_kwargs,
                            )

                            multitask_fulldgl_nn = FullGraphMultiTaskNetwork(
                                task_heads_kwargs=task_heads_params,
                                gnn_kwargs=gnn_kwargs,
                                pre_nn_kwargs=pre_nn_kwargs,
                                post_nn_kwargs=post_nn_kwargs,
                            )

                            bg = deepcopy(self.bg)
                            h_out = multitask_fulldgl_nn.forward(bg)

                            dim_1 = bg.num_nodes() if pooling == ["none"] else bg.batch_size
                            #self.assertListEqual(list(h_out.shape), [dim_1, self.out_dim], msg=err_msg)

                            self.assertListEqual(list(h_out[task_1_kwargs["task_name"]].shape), [dim_1, task_1_kwargs["out_dim"]], msg=err_msg)
                            self.assertListEqual(list(h_out[task_2_kwargs["task_name"]].shape), [dim_1, task_2_kwargs["out_dim"]], msg=err_msg)
                            self.assertListEqual(list(h_out[task_3_kwargs["task_name"]].shape), [dim_1, task_3_kwargs["out_dim"]], msg=err_msg)


class test_FullGraphMultiTaskNetwork(ut.TestCase):

    in_dim_nodes = 7
    in_dim_edges = 13

    g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
    g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
    g1.ndata["feat"] = torch.zeros(g1.num_nodes(), in_dim_nodes, dtype=torch.float32)
    g1.edata["feat"] = torch.ones(g1.num_edges(), in_dim_edges, dtype=torch.float32)
    g2.ndata["feat"] = torch.ones(g2.num_nodes(), in_dim_nodes, dtype=torch.float32)
    g2.edata["feat"] = torch.zeros(g2.num_edges(), in_dim_edges, dtype=torch.float32)
    batch = [g1, g2, deepcopy(g1), deepcopy(g2)]
    batch = [dgl.add_self_loop(g) for g in batch]
    bg = dgl.batch(batch)

    def test_FullGraphMultiTaskNetwork_from_config(self):
        cfg = goli.load_config(name="zinc_default_multitask_fulldgl")

        # Initialize the network
        model_class, model_kwargs = load_architecture(
            cfg,
            in_dim_nodes=self.in_dim_nodes,
            in_dim_edges=self.in_dim_edges,
        )

        multitask_fulldgl_nn = model_class(**model_kwargs)

        # Test
        bg = deepcopy(self.bg)
        h_out = multitask_fulldgl_nn.forward(bg)

        dim_1 = self.bg.batch_size

        self.assertListEqual(list(h_out[task_1_kwargs["task_name"]].shape), [dim_1, task_1_kwargs["out_dim"]])
        self.assertListEqual(list(h_out[task_2_kwargs["task_name"]].shape), [dim_1, task_2_kwargs["out_dim"]])
        self.assertListEqual(list(h_out[task_3_kwargs["task_name"]].shape), [dim_1, task_3_kwargs["out_dim"]])


if __name__ == "__main__":
    ut.main()
