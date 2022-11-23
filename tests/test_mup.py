"""
Unit tests for the implementation of mup
"""

import unittest as ut
from copy import deepcopy
import torch.nn as nn
import torch
import yaml

from torch_geometric.data import Batch, Data

from goli.nn.architectures import FeedForwardNN, FeedForwardPyg, FullGraphNetwork, FullGraphMultiTaskNetwork


def get_pyg_graphs(in_dim, in_dim_edges):
    edge_idx1 = torch.stack([torch.tensor([0, 1, 2, 3, 2]), torch.tensor([1, 2, 3, 0, 0])])
    edge_idx2 = torch.stack([torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])])
    x1 = torch.randn(edge_idx1.max() + 1, in_dim, dtype=torch.float32)
    e1 = torch.randn(edge_idx1.shape[-1], in_dim_edges, dtype=torch.float32)
    x2 = torch.randn(edge_idx2.max() + 1, in_dim, dtype=torch.float32)
    e2 = torch.randn(edge_idx2.shape[-1], in_dim_edges, dtype=torch.float32)
    g1 = Data(h=x1, edge_index=edge_idx1, edge_attr=e1)
    g2 = Data(h=x2, edge_index=edge_idx2, edge_attr=e2)
    bg = Batch.from_data_list([g1, g2])

    return bg


class test_mup(ut.TestCase):
    kwargs = dict(
        in_dim=12,
        out_dim=60,
        hidden_dims=8 * [84],
        depth=None,
        activation="LeakyReLU",
        last_activation="LeakyReLU",
        dropout=0.1,
        last_dropout=0.2,
        normalization="batch_norm",
        first_normalization="batch_norm",
        last_normalization="batch_norm",
        residual_type="simple",
        residual_skip_steps=2,
        name="testing",
        layer_type="fc",
        layer_kwargs=None,
    )

    def test_feedforwardnn_mup(self):
        kwargs = deepcopy(self.kwargs)
        model = FeedForwardNN(**kwargs, last_layer_is_readout=False)
        model_lastreadout = FeedForwardNN(**kwargs, last_layer_is_readout=True)
        base_1 = model.make_mup_base_kwargs(divide_factor=1)
        base_1_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=1)

        base_2 = model.make_mup_base_kwargs(divide_factor=2)
        base_2_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=2)
        kwargs_2 = deepcopy(base_1)
        kwargs_2.update(dict(out_dim=30, hidden_dims=8 * [42]))
        kwargs_2_lastreadout = deepcopy(base_1_lastreadout)
        kwargs_2_lastreadout.update(dict(hidden_dims=8 * [42]))

        # Check the kwargs matching
        for key in kwargs_2.keys():
            if isinstance(kwargs_2[key], nn.Module):
                # Can't match the random weights
                self.assertEqual(str(kwargs_2[key]), str(base_2[key]), msg=key)
            else:
                self.assertEqual(kwargs_2[key], base_2[key], msg=key)

        # Check the kwargs matching
        for key in kwargs_2_lastreadout.keys():
            if isinstance(kwargs_2_lastreadout[key], nn.Module):
                # Can't match the random weights
                self.assertEqual(str(kwargs_2_lastreadout[key]), str(base_2_lastreadout[key]), msg=key)
            else:
                self.assertEqual(kwargs_2_lastreadout[key], base_2_lastreadout[key], msg=key)

        # Test that the models with divide_factor=1 can be built run a forward pass
        in_features = torch.randn((10, kwargs["in_dim"]))
        model_1 = FeedForwardNN(**base_1)
        model_1.forward(deepcopy(in_features))
        model_1_lastreadout = FeedForwardNN(**base_1_lastreadout)
        model_1_lastreadout.forward(deepcopy(in_features))

        # Test that the models with divide_factor=2 can be built run a forward pass
        model_2 = FeedForwardNN(**base_2)
        model_2.forward(deepcopy(in_features))
        model_2_lastreadout = FeedForwardNN(**base_2_lastreadout)
        model_2_lastreadout.forward(deepcopy(in_features))


    def test_feedforwardgraph_mup(self):
        kwargs = deepcopy(self.kwargs)
        in_dim_edges = kwargs["in_dim"]
        kwargs.update(dict(layer_type="pyg:gine", in_dim_edges=in_dim_edges))
        model = FeedForwardPyg(**kwargs, last_layer_is_readout=False)
        model_lastreadout = FeedForwardPyg(**kwargs, last_layer_is_readout=True)
        base_1 = model.make_mup_base_kwargs(divide_factor=1)
        base_1_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=1)

        base_2 = model.make_mup_base_kwargs(divide_factor=2)
        base_2_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=2)
        kwargs_2 = deepcopy(base_1)
        kwargs_2.update(dict(out_dim=30, hidden_dims=8 * [42]))
        kwargs_2_lastreadout = deepcopy(base_1_lastreadout)
        kwargs_2_lastreadout.update(dict(hidden_dims=8 * [42]))

        # Check the kwargs matching
        for key in kwargs_2.keys():
            if isinstance(kwargs_2[key], nn.Module):
                # Can't match the random weights
                self.assertEqual(str(kwargs_2[key]), str(base_2[key]), msg=key)
            else:
                self.assertEqual(kwargs_2[key], base_2[key], msg=key)

        # Check the kwargs matching
        for key in kwargs_2_lastreadout.keys():
            if isinstance(kwargs_2_lastreadout[key], nn.Module):
                # Can't match the random weights
                self.assertEqual(str(kwargs_2_lastreadout[key]), str(base_2_lastreadout[key]), msg=key)
            else:
                self.assertEqual(kwargs_2_lastreadout[key], base_2_lastreadout[key], msg=key)

        # Test that the models with divide_factor=1 can be built run a forward pass
        in_features = get_pyg_graphs(in_dim=kwargs["in_dim"], in_dim_edges=in_dim_edges)
        model_1 = FeedForwardPyg(**base_1)
        model_1.forward(deepcopy(in_features))
        model_1_lastreadout = FeedForwardPyg(**base_1_lastreadout)
        model_1_lastreadout.forward(deepcopy(in_features))

        # Test that the models with divide_factor=2 can be built run a forward pass
        model_2 = FeedForwardPyg(**base_2)
        model_2.forward(deepcopy(in_features))
        model_2_lastreadout = FeedForwardPyg(**base_2_lastreadout)
        model_2_lastreadout.forward(deepcopy(in_features))


    def test_fullgraphnetwork(self):

        # Load the configuration file for the model
        CONFIG_FILE = "tests/config_test_ipu_dataloader.yaml"
        with open(CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)

        # Make fake graphs
        in_dim = 12
        in_dim_edges = 12
        pe_indims = {"rw_pos/rwse": 16,
                    "la_pos/eigvecs": 3,
                    "la_pos/eigvals": 3}
        in_features = get_pyg_graphs(in_dim=in_dim, in_dim_edges=in_dim_edges)
        in_features["feat"] = in_features["h"]
        in_features["edge_feat"] = in_features["edge_attr"]
        for key, dim in pe_indims.items():
            in_features[key] = torch.randn(in_features.num_nodes, dim)


        # Load the model
        kwargs = {}
        for key, val in cfg["architecture"].items():
            if key in ["model_type", "task_heads", "mup_base_path"]:
                continue
            kwargs[key + "_kwargs"] = val
        kwargs["pre_nn_kwargs"]["in_dim"] = in_dim + kwargs["pe_encoders_kwargs"]["out_dim"]
        kwargs["pre_nn_edges_kwargs"]["in_dim"] = in_dim_edges
        kwargs["pe_encoders_kwargs"]["in_dims"] = pe_indims

        model = FullGraphNetwork(**kwargs, last_layer_is_readout=True)

        kw_1 = model.make_mup_base_kwargs(divide_factor=1)
        kw_2 = model.make_mup_base_kwargs(divide_factor=2)

        # Check the parameter sizes
        for key, elem in kw_1.items():
            if not isinstance(elem, dict):
                continue
            for subkey, subelem in elem.items():
                if "dim" in subkey:
                    match = f"{key}:{subkey}"
                    if match in [
                        "pre_nn_edges_kwargs:in_dim",
                        "post_nn_kwargs:out_dim",
                    ]:
                        # Constants
                        self.assertEqual(subelem, kw_2[key][subkey], msg=match)
                    elif match in ["pre_nn_kwargs:in_dim"]:
                        self.assertNotEqual(subelem, kw_2[key][subkey], msg=match)
                    elif match in [
                        "pre_nn_kwargs:out_dim",
                        "pre_nn_edges_kwargs:out_dim",
                        "gnn_kwargs:in_dim",
                        "post_nn_kwargs:in_dim",
                        "gnn_kwargs:out_dim",
                        "gnn_kwargs:in_dim_edges",
                        "pe_encoders_kwargs:out_dim",
                    ]:
                        # Divide by 2
                        self.assertEqual(round(subelem / 2), kw_2[key][subkey], msg=match)
                    elif match in [
                        "pre_nn_kwargs:hidden_dims",
                        "pre_nn_edges_kwargs:hidden_dims",
                        "gnn_kwargs:hidden_dims",
                        "post_nn_kwargs:hidden_dims",
                        "gnn_kwargs:hidden_dims_edges",
                    ]:
                        # Arrays divide by 2
                        new_list = [round(e / 2) for e in subelem]
                        self.assertListEqual(new_list, kw_2[key][subkey], msg=match)
                    else:
                        print(match)

        # Test that the models with divide_factor=1 can be built run a forward pass
        kw_1["last_layer_is_readout"] = False
        model_1 = FullGraphNetwork(**kw_1)
        model_1.forward(deepcopy(in_features))
        kw_1["last_layer_is_readout"] = True
        model_1 = FullGraphNetwork(**kw_1)
        model_1.forward(deepcopy(in_features))


        # Test that the models with divide_factor=2 can be built run a forward pass
        kw_2["last_layer_is_readout"] = False
        model_2 = FullGraphNetwork(**kw_2)
        model_2.forward(deepcopy(in_features))
        kw_2["last_layer_is_readout"] = True
        model_2 = FullGraphNetwork(**kw_2)
        model_2.forward(deepcopy(in_features))




    def test_fullgraphmultitasknetwork(self):

        # Load the configuration file for the model
        CONFIG_FILE = "tests/config_test_ipu_dataloader.yaml"
        with open(CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)

        # Make fake graphs
        in_dim = 12
        in_dim_edges = 12
        pe_indims = {"rw_pos/rwse": 16,
                    "la_pos/eigvecs": 3,
                    "la_pos/eigvals": 3}
        in_features = get_pyg_graphs(in_dim=in_dim, in_dim_edges=in_dim_edges)
        in_features["feat"] = in_features["h"]
        in_features["edge_feat"] = in_features["edge_attr"]
        for key, dim in pe_indims.items():
            in_features[key] = torch.randn(in_features.num_nodes, dim)

        # Load the model
        kwargs = {}
        for key, val in cfg["architecture"].items():
            if key in ["model_type", "mup_base_path"]:
                continue
            kwargs[key + "_kwargs"] = val
        kwargs["pre_nn_kwargs"]["in_dim"] = in_dim + kwargs["pe_encoders_kwargs"]["out_dim"]
        kwargs["pre_nn_edges_kwargs"]["in_dim"] = in_dim_edges
        kwargs["pe_encoders_kwargs"]["in_dims"] = pe_indims

        model = FullGraphMultiTaskNetwork(**kwargs, last_layer_is_readout=True)

        kw_1 = model.make_mup_base_kwargs(divide_factor=1)
        kw_2 = model.make_mup_base_kwargs(divide_factor=2)

        # Check the parameter sizes
        for key, elem in kw_1.items():
            if not isinstance(elem, dict):
                continue
            for subkey, subelem in elem.items():
                if "dim" in subkey:
                    match = f"{key}:{subkey}"
                    if match in ["pre_nn_edges_kwargs:in_dim"]:
                        # Constants
                        self.assertEqual(subelem, kw_2[key][subkey], msg=match)
                    elif match in ["pre_nn_kwargs:in_dim"]:
                        self.assertNotEqual(subelem, kw_2[key][subkey], msg=match)
                    elif match in [
                        "pre_nn_kwargs:out_dim",
                        "pre_nn_edges_kwargs:out_dim",
                        "gnn_kwargs:in_dim",
                        "post_nn_kwargs:in_dim",
                        "gnn_kwargs:out_dim",
                        "gnn_kwargs:in_dim_edges",
                        "pe_encoders_kwargs:out_dim",
                        "post_nn_kwargs:out_dim",
                    ]:
                        # Divide by 2
                        self.assertEqual(round(subelem / 2), kw_2[key][subkey], msg=match)
                    elif match in [
                        "pre_nn_kwargs:hidden_dims",
                        "pre_nn_edges_kwargs:hidden_dims",
                        "gnn_kwargs:hidden_dims",
                        "post_nn_kwargs:hidden_dims",
                        "gnn_kwargs:hidden_dims_edges",
                    ]:
                        # Arrays divide by 2
                        new_list = [round(e / 2) for e in subelem]
                        self.assertListEqual(new_list, kw_2[key][subkey], msg=match)
                elif subkey in ["homo", "alpha", "cv"]:
                    for subsubkey, subsubelem in subelem.items():
                        match = f"{key}:{subsubkey}"
                        if match in [
                            "task_heads_kwargs:out_dim",
                            "task_heads_kwargs:out_dim",
                            "task_heads_kwargs:out_dim",
                        ]:
                            # No divide
                            self.assertEqual(subsubelem, kw_2[key][subkey][subsubkey], msg=match)
                        elif match in [
                            "task_heads_kwargs:in_dim",
                            "task_heads_kwargs:in_dim",
                            "task_heads_kwargs:in_dim",
                        ]:
                            # Divide by 2
                            self.assertEqual(round(subsubelem / 2), kw_2[key][subkey][subsubkey], msg=match)
                        elif match in [
                            "task_heads_kwargs:hidden_dims",
                            "task_heads_kwargs:hidden_dims",
                            "task_heads_kwargs:hidden_dims",
                        ]:
                            # Divide by 2 a list
                            new_list = [round(e / 2) for e in subsubelem]
                            self.assertListEqual(new_list, kw_2[key][subkey][subsubkey], msg=match)

        # Test that the models with divide_factor=1 can be built run a forward pass
        kw_1["last_layer_is_readout"] = False
        model_1 = FullGraphMultiTaskNetwork(**kw_1)
        model_1.forward(deepcopy(in_features))
        kw_1["last_layer_is_readout"] = True
        model_1 = FullGraphMultiTaskNetwork(**kw_1)
        model_1.forward(deepcopy(in_features))


        # Test that the models with divide_factor=2 can be built run a forward pass
        kw_2["last_layer_is_readout"] = False
        model_2 = FullGraphMultiTaskNetwork(**kw_2)
        model_2.forward(deepcopy(in_features))
        kw_2["last_layer_is_readout"] = True
        model_2 = FullGraphMultiTaskNetwork(**kw_2)
        model_2.forward(deepcopy(in_features))



if __name__ == "__main__":
    ut.main()
