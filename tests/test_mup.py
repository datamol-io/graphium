"""
Unit tests for the implementation of mup
"""

import unittest as ut
from copy import deepcopy
import torch.nn as nn
import yaml

from goli.nn.architectures import FeedForwardNN, FeedForwardPyg, FullGraphNetwork, FullGraphMultiTaskNetwork

class test_nan_statistics(ut.TestCase):
    kwargs = dict(
            in_dim=12,
            out_dim=60,
            hidden_dims=8*[84],
            depth=None,
            activation = "LeakyReLU",
            last_activation = "LeakyReLU",
            dropout = 0.1,
            last_dropout = 0.2,
            normalization = "batch_norm",
            first_normalization = "batch_norm",
            last_normalization = "batch_norm",
            residual_type = "simple",
            residual_skip_steps = 2,
            name = "testing",
            layer_type = "fc",
            layer_kwargs = None,
        )

    def test_feedforwardnn_mup(self):
        kwargs = deepcopy(self.kwargs)
        model = FeedForwardNN(**kwargs, last_layer_is_readout = False)
        model_lastreadout = FeedForwardNN(**kwargs, last_layer_is_readout = True)
        base_1 = model.make_mup_base_kwargs(divide_factor=1)
        base_1_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=1)

        base_2 = model.make_mup_base_kwargs(divide_factor=2)
        base_2_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=2)
        kwargs_2 = deepcopy(base_1)
        kwargs_2.update(dict(out_dim=30, hidden_dims=8*[42]))
        kwargs_2_lastreadout = deepcopy(base_1_lastreadout)
        kwargs_2_lastreadout.update(dict(hidden_dims=8*[42]))

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


    def test_feedforwardgrapg_mup(self):
        kwargs = deepcopy(self.kwargs)
        kwargs.update(dict(
            layer_type="pyg:gin"
        ))
        model = FeedForwardPyg(**kwargs, last_layer_is_readout = False)
        model_lastreadout = FeedForwardPyg(**kwargs, last_layer_is_readout = True)
        base_1 = model.make_mup_base_kwargs(divide_factor=1)
        base_1_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=1)

        base_2 = model.make_mup_base_kwargs(divide_factor=2)
        base_2_lastreadout = model_lastreadout.make_mup_base_kwargs(divide_factor=2)
        kwargs_2 = deepcopy(base_1)
        kwargs_2.update(dict(out_dim=30, hidden_dims=8*[42]))
        kwargs_2_lastreadout = deepcopy(base_1_lastreadout)
        kwargs_2_lastreadout.update(dict(hidden_dims=8*[42]))

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


    def test_fullgraphnetwork(self):

        # Load the configuration file for the model
        CONFIG_FILE = "tests/config_test_ipu_dataloader.yaml"
        with open(CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)

        # Load the model
        kwargs = {}
        for key, val in cfg["architecture"].items():
            if key in ["model_type", "task_heads", "mup_base_path"]:
                continue
            kwargs[key + "_kwargs"] = val
        kwargs["pre_nn_kwargs"]["in_dim"] = 5
        kwargs["pre_nn_edges_kwargs"]["in_dim"] = 7
        kwargs["pe_encoders_kwargs"]["in_dims"] = {'rw_pos/rwse': 16, 'la_pos/eigvecs': 3, 'la_pos/eigvals': 3}
        model = FullGraphNetwork(**kwargs, last_layer_is_readout=True)

        kw_1 = model.make_mup_base_kwargs(divide_factor=1)
        kw_2 = model.make_mup_base_kwargs(divide_factor=2)

        for key, elem in kw_1.items():
            if not isinstance(elem, dict):
                continue
            for subkey, subelem in elem.items():
                if "dim" in subkey:
                    match = f"{key}:{subkey}"
                    if match in ["pre_nn_kwargs:in_dim", "pre_nn_edges_kwargs:in_dim", "post_nn_kwargs:out_dim"]:
                        # Constants
                        self.assertEqual(subelem, kw_2[key][subkey], msg=match)
                    elif match in ["pre_nn_kwargs:out_dim", "pre_nn_edges_kwargs:out_dim", "gnn_kwargs:in_dim", "post_nn_kwargs:in_dim", "gnn_kwargs:out_dim", "gnn_kwargs:in_dim_edges", "pe_encoders_kwargs:out_dim"]:
                        # Divide by 2
                        self.assertEqual(round(subelem/2), kw_2[key][subkey], msg=match)
                    elif match in ["pre_nn_kwargs:hidden_dims", "pre_nn_edges_kwargs:hidden_dims", "gnn_kwargs:hidden_dims", "post_nn_kwargs:hidden_dims", "gnn_kwargs:hidden_dims_edges"]:
                        # Arrays divide by 2
                        new_list = [round(e/2) for e in subelem]
                        self.assertListEqual(new_list, kw_2[key][subkey], msg=match)
                    else:
                        print(match)


    def test_fullgraphmultitasknetwork(self):

        # Load the configuration file for the model
        CONFIG_FILE = "tests/config_test_ipu_dataloader.yaml"
        with open(CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)

        # Load the model
        kwargs = {}
        for key, val in cfg["architecture"].items():
            if key in ["model_type", "mup_base_path"]:
                continue
            kwargs[key + "_kwargs"] = val
        kwargs["pre_nn_kwargs"]["in_dim"] = 5
        kwargs["pre_nn_edges_kwargs"]["in_dim"] = 7
        kwargs["pe_encoders_kwargs"]["in_dims"] = {'rw_pos/rwse': 16, 'la_pos/eigvecs': 3, 'la_pos/eigvals': 3}
        model = FullGraphMultiTaskNetwork(**kwargs, last_layer_is_readout=True)

        kw_1 = model.make_mup_base_kwargs(divide_factor=1)
        kw_2 = model.make_mup_base_kwargs(divide_factor=2)

        for key, elem in kw_1.items():
            if not isinstance(elem, dict):
                continue
            for subkey, subelem in elem.items():
                if "dim" in subkey:
                    match = f"{key}:{subkey}"
                    if match in ["pre_nn_kwargs:in_dim", "pre_nn_edges_kwargs:in_dim"]:
                        # Constants
                        self.assertEqual(subelem, kw_2[key][subkey], msg=match)
                    elif match in ["pre_nn_kwargs:out_dim", "pre_nn_edges_kwargs:out_dim", "gnn_kwargs:in_dim", "post_nn_kwargs:in_dim", "gnn_kwargs:out_dim", "gnn_kwargs:in_dim_edges", "pe_encoders_kwargs:out_dim", "post_nn_kwargs:out_dim"]:
                        # Divide by 2
                        self.assertEqual(round(subelem/2), kw_2[key][subkey], msg=match)
                    elif match in ["pre_nn_kwargs:hidden_dims", "pre_nn_edges_kwargs:hidden_dims", "gnn_kwargs:hidden_dims", "post_nn_kwargs:hidden_dims", "gnn_kwargs:hidden_dims_edges"]:
                        # Arrays divide by 2
                        new_list = [round(e/2) for e in subelem]
                        self.assertListEqual(new_list, kw_2[key][subkey], msg=match)
                elif subkey in ["homo", "alpha", "cv"]:
                    for subsubkey, subsubelem in subelem.items():
                        match = f"{key}:{subsubkey}"
                        if match in ["task_heads_kwargs:out_dim", "task_heads_kwargs:out_dim", "task_heads_kwargs:out_dim"]:
                            # No divide
                            self.assertEqual(subsubelem, kw_2[key][subkey][subsubkey], msg=match)
                        elif match in ["task_heads_kwargs:in_dim", "task_heads_kwargs:in_dim", "task_heads_kwargs:in_dim"]:
                            # Divide by 2
                            self.assertEqual(round(subsubelem/2), kw_2[key][subkey][subsubkey], msg=match)
                        elif match in ["task_heads_kwargs:hidden_dims", "task_heads_kwargs:hidden_dims", "task_heads_kwargs:hidden_dims"]:
                            # Divide by 2 a list
                            new_list = [round(e/2) for e in subsubelem]
                            self.assertListEqual(new_list, kw_2[key][subkey][subsubkey], msg=match)


if __name__ == "__main__":
    ut.main()
