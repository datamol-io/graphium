"""
Unit tests for the different layers of graphium/nn/ensemble_layers
"""

import numpy as np
import torch
from torch.nn import Linear
import unittest as ut

from graphium.nn.base_layers import FCLayer, MLP, MuReadoutGraphium
from graphium.nn.ensemble_layers import (
    EnsembleLinear,
    EnsembleFCLayer,
    EnsembleMLP,
    EnsembleMuReadoutGraphium,
)
from graphium.nn.architectures import FeedForwardNN, EnsembleFeedForwardNN


class test_Ensemble_Layers(ut.TestCase):
    def check_ensemble_linear(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        batch_size: int,
        more_batch_dim: int,
        use_mureadout=False,
    ):
        msg = f"Testing EnsembleLinear with in_dim={in_dim}, out_dim={out_dim}, num_ensemble={num_ensemble}, batch_size={batch_size}, more_batch_dim={more_batch_dim}"

        if use_mureadout:
            # Create EnsembleMuReadoutGraphium instance
            ensemble_linear = EnsembleMuReadoutGraphium(in_dim, out_dim, num_ensemble)
            # Create equivalent separate Linear layers with synchronized weights and biases
            linear_layers = [MuReadoutGraphium(in_dim, out_dim) for _ in range(num_ensemble)]
        else:
            # Create EnsembleLinear instance
            ensemble_linear = EnsembleLinear(in_dim, out_dim, num_ensemble)
            # Create equivalent separate Linear layers with synchronized weights and biases
            linear_layers = [Linear(in_dim, out_dim) for _ in range(num_ensemble)]

        for i, linear_layer in enumerate(linear_layers):
            linear_layer.weight.data = ensemble_linear.weight.data[i]
            if ensemble_linear.bias is not None:
                linear_layer.bias.data = ensemble_linear.bias.data[i].squeeze()

        # Test with a sample input
        input_tensor = torch.randn(batch_size, in_dim)
        ensemble_output = ensemble_linear(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, (num_ensemble, batch_size, out_dim), msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, linear_layer in enumerate(linear_layers):
            individual_output = linear_layer(input_tensor)
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output[i].detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

        # Test with a sample input with the extra `num_ensemble` and `more_batch_dim` dimension
        if more_batch_dim:
            out_shape = (more_batch_dim, num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(more_batch_dim, num_ensemble, batch_size, in_dim)
        else:
            out_shape = (num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(num_ensemble, batch_size, in_dim)
        ensemble_output = ensemble_linear(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, out_shape, msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, linear_layer in enumerate(linear_layers):
            if more_batch_dim:
                individual_output = linear_layer(input_tensor[:, i])
                ensemble_output_i = ensemble_output[:, i]
            else:
                individual_output = linear_layer(input_tensor[i])
                ensemble_output_i = ensemble_output[i]
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output_i.detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

    def test_ensemble_linear(self):
        # more_batch_dim=0
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0)
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=0)
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=0)

        # more_batch_dim=1
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1)
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=1)
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=1)

        # more_batch_dim=7
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7)
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=7)
        self.check_ensemble_linear(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=7)

    def test_ensemble_mureadout_graphium(self):
        # Test `use_mureadout`
        # more_batch_dim=0
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, use_mureadout=True
        )
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=0, use_mureadout=True
        )
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=0, use_mureadout=True
        )

        # more_batch_dim=1
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, use_mureadout=True
        )
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=1, use_mureadout=True
        )
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=1, use_mureadout=True
        )

        # more_batch_dim=7
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, use_mureadout=True
        )
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=7, use_mureadout=True
        )
        self.check_ensemble_linear(
            in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=7, use_mureadout=True
        )

    def check_ensemble_fclayer(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        batch_size: int,
        more_batch_dim: int,
        is_readout_layer=False,
    ):
        msg = f"Testing EnsembleFCLayer with in_dim={in_dim}, out_dim={out_dim}, num_ensemble={num_ensemble}, batch_size={batch_size}, more_batch_dim={more_batch_dim}"

        # Create EnsembleFCLayer instance
        ensemble_fclayer = EnsembleFCLayer(in_dim, out_dim, num_ensemble, is_readout_layer=is_readout_layer)

        # Create equivalent separate FCLayer layers with synchronized weights and biases
        fc_layers = [FCLayer(in_dim, out_dim, is_readout_layer=is_readout_layer) for _ in range(num_ensemble)]
        for i, fc_layer in enumerate(fc_layers):
            fc_layer.linear.weight.data = ensemble_fclayer.linear.weight.data[i]
            if ensemble_fclayer.bias is not None:
                fc_layer.linear.bias.data = ensemble_fclayer.linear.bias.data[i].squeeze()

        # Test with a sample input
        input_tensor = torch.randn(batch_size, in_dim)
        ensemble_output = ensemble_fclayer(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, (num_ensemble, batch_size, out_dim), msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, fc_layer in enumerate(fc_layers):
            individual_output = fc_layer(input_tensor)
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output[i].detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

        # Test with a sample input with the extra `num_ensemble` and `more_batch_dim` dimension
        if more_batch_dim:
            out_shape = (more_batch_dim, num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(more_batch_dim, num_ensemble, batch_size, in_dim)
        else:
            out_shape = (num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(num_ensemble, batch_size, in_dim)
        ensemble_output = ensemble_fclayer(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, out_shape, msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, fc_layer in enumerate(fc_layers):
            if more_batch_dim:
                individual_output = fc_layer(input_tensor[:, i])
                ensemble_output_i = ensemble_output[:, i]
            else:
                individual_output = fc_layer(input_tensor[i])
                ensemble_output_i = ensemble_output[i]
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output_i.detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

    def test_ensemble_fclayer(self):
        # more_batch_dim=0
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0)
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=0)
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=0)

        # more_batch_dim=1
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1)
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=1)
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=1)

        # more_batch_dim=7
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7)
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=7)
        self.check_ensemble_fclayer(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=7)

        # Test `is_readout_layer`
        self.check_ensemble_fclayer(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, is_readout_layer=True
        )
        self.check_ensemble_fclayer(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, is_readout_layer=True
        )
        self.check_ensemble_fclayer(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, is_readout_layer=True
        )

    def check_ensemble_mlp(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        batch_size: int,
        more_batch_dim: int,
        last_layer_is_readout=False,
    ):
        msg = f"Testing EnsembleMLP with in_dim={in_dim}, out_dim={out_dim}, num_ensemble={num_ensemble}, batch_size={batch_size}, more_batch_dim={more_batch_dim}"

        # Create EnsembleMLP instance
        hidden_dims = [17, 17, 17]
        ensemble_mlp = EnsembleMLP(
            in_dim, hidden_dims, out_dim, num_ensemble, last_layer_is_readout=last_layer_is_readout
        )

        # Create equivalent separate MLP layers with synchronized weights and biases
        mlps = [
            MLP(in_dim, hidden_dims, out_dim, last_layer_is_readout=last_layer_is_readout)
            for _ in range(num_ensemble)
        ]
        for i, mlp in enumerate(mlps):
            for j, layer in enumerate(mlp.fully_connected):
                layer.linear.weight.data = ensemble_mlp.fully_connected[j].linear.weight.data[i]
                if layer.bias is not None:
                    layer.linear.bias.data = ensemble_mlp.fully_connected[j].linear.bias.data[i].squeeze()

        # Test with a sample input
        input_tensor = torch.randn(batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, (num_ensemble, batch_size, out_dim), msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, mlp in enumerate(mlps):
            individual_output = mlp(input_tensor)
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output[i].detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

        # Test with a sample input with the extra `num_ensemble` and `more_batch_dim` dimension
        if more_batch_dim:
            out_shape = (more_batch_dim, num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(more_batch_dim, num_ensemble, batch_size, in_dim)
        else:
            out_shape = (num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(num_ensemble, batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, out_shape, msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, mlp in enumerate(mlps):
            if more_batch_dim:
                individual_output = mlp(input_tensor[:, i])
                ensemble_output_i = ensemble_output[:, i]
            else:
                individual_output = mlp(input_tensor[i])
                ensemble_output_i = ensemble_output[i]
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output_i.detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

    def test_ensemble_mlp(self):
        # more_batch_dim=0
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0)
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=0)
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=0)

        # more_batch_dim=1
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1)
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=1)
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=1)

        # more_batch_dim=7
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7)
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=7)
        self.check_ensemble_mlp(in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=7)

        # Test `last_layer_is_readout`
        self.check_ensemble_mlp(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, last_layer_is_readout=True
        )
        self.check_ensemble_mlp(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, last_layer_is_readout=True
        )
        self.check_ensemble_mlp(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, last_layer_is_readout=True
        )

    def check_ensemble_feedforwardnn(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        batch_size: int,
        more_batch_dim: int,
        last_layer_is_readout=False,
    ):
        msg = f"Testing EnsembleFeedForwardNN with in_dim={in_dim}, out_dim={out_dim}, num_ensemble={num_ensemble}, batch_size={batch_size}, more_batch_dim={more_batch_dim}"

        # Create EnsembleFeedForwardNN instance
        hidden_dims = [17, 17, 17]
        ensemble_mlp = EnsembleFeedForwardNN(
            in_dim,
            out_dim,
            hidden_dims,
            num_ensemble,
            reduction=None,
            last_layer_is_readout=last_layer_is_readout,
        )

        # Create equivalent separate MLP layers with synchronized weights and biases
        mlps = [
            FeedForwardNN(in_dim, out_dim, hidden_dims, last_layer_is_readout=last_layer_is_readout)
            for _ in range(num_ensemble)
        ]
        for i, mlp in enumerate(mlps):
            for j, layer in enumerate(mlp.layers):
                layer.linear.weight.data = ensemble_mlp.layers[j].linear.weight.data[i]
                if layer.bias is not None:
                    layer.linear.bias.data = ensemble_mlp.layers[j].linear.bias.data[i].squeeze()

        # Test with a sample input
        input_tensor = torch.randn(batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, (num_ensemble, batch_size, out_dim), msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        individual_outputs = []
        for i, mlp in enumerate(mlps):
            individual_outputs.append(mlp(input_tensor))
        individual_outputs = torch.stack(individual_outputs).detach().numpy()
        for i, mlp in enumerate(mlps):
            ensemble_output_i = ensemble_output[i].detach().numpy()
            np.testing.assert_allclose(
                ensemble_output_i, individual_outputs[..., i, :, :], atol=1e-5, err_msg=msg
            )

        # Test with a sample input with the extra `num_ensemble` and `more_batch_dim` dimension
        if more_batch_dim:
            out_shape = (more_batch_dim, num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(more_batch_dim, num_ensemble, batch_size, in_dim)
        else:
            out_shape = (num_ensemble, batch_size, out_dim)
            input_tensor = torch.randn(num_ensemble, batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, out_shape, msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        for i, mlp in enumerate(mlps):
            if more_batch_dim:
                individual_output = mlp(input_tensor[:, i])
                ensemble_output_i = ensemble_output[:, i]
            else:
                individual_output = mlp(input_tensor[i])
                ensemble_output_i = ensemble_output[i]
            individual_output = individual_output.detach().numpy()
            ensemble_output_i = ensemble_output_i.detach().numpy()
            np.testing.assert_allclose(ensemble_output_i, individual_output, atol=1e-5, err_msg=msg)

    def check_ensemble_feedforwardnn_mean(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        batch_size: int,
        more_batch_dim: int,
        last_layer_is_readout=False,
    ):
        msg = f"Testing EnsembleFeedForwardNN with in_dim={in_dim}, out_dim={out_dim}, num_ensemble={num_ensemble}, batch_size={batch_size}, more_batch_dim={more_batch_dim}"

        # Create EnsembleFeedForwardNN instance
        hidden_dims = [17, 17, 17]
        ensemble_mlp = EnsembleFeedForwardNN(
            in_dim,
            out_dim,
            hidden_dims,
            num_ensemble,
            reduction="mean",
            last_layer_is_readout=last_layer_is_readout,
        )

        # Create equivalent separate MLP layers with synchronized weights and biases
        mlps = [
            FeedForwardNN(in_dim, out_dim, hidden_dims, last_layer_is_readout=last_layer_is_readout)
            for _ in range(num_ensemble)
        ]
        for i, mlp in enumerate(mlps):
            for j, layer in enumerate(mlp.layers):
                layer.linear.weight.data = ensemble_mlp.layers[j].linear.weight.data[i]
                if layer.bias is not None:
                    layer.linear.bias.data = ensemble_mlp.layers[j].linear.bias.data[i].squeeze()

        # Test with a sample input
        input_tensor = torch.randn(batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, (batch_size, out_dim), msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        individual_outputs = []
        for i, mlp in enumerate(mlps):
            individual_outputs.append(mlp(input_tensor))
        individual_outputs = torch.stack(individual_outputs, dim=-3)
        individual_outputs = individual_outputs.mean(dim=-3).detach().numpy()
        np.testing.assert_allclose(
            ensemble_output.detach().numpy(), individual_outputs, atol=1e-5, err_msg=msg
        )

        # Test with a sample input with the extra `num_ensemble` and `more_batch_dim` dimension
        if more_batch_dim:
            out_shape = (more_batch_dim, batch_size, out_dim)
            input_tensor = torch.randn(more_batch_dim, num_ensemble, batch_size, in_dim)
        else:
            out_shape = (batch_size, out_dim)
            input_tensor = torch.randn(num_ensemble, batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor).detach()

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, out_shape, msg=msg)

        # Make sure that the outputs of the individual layers are the same as the ensemble output
        individual_outputs = []
        for i, mlp in enumerate(mlps):
            if more_batch_dim:
                individual_outputs.append(mlp(input_tensor[:, i]))
            else:
                individual_outputs.append(mlp(input_tensor[i]))
        individual_output = torch.stack(individual_outputs, dim=-3).mean(dim=-3).detach().numpy()
        np.testing.assert_allclose(ensemble_output, individual_output, atol=1e-5, err_msg=msg)

    def check_ensemble_feedforwardnn_simple(
        self,
        in_dim: int,
        out_dim: int,
        num_ensemble: int,
        batch_size: int,
        more_batch_dim: int,
        last_layer_is_readout=False,
        **kwargs,
    ):
        msg = f"Testing EnsembleFeedForwardNN with in_dim={in_dim}, out_dim={out_dim}, num_ensemble={num_ensemble}, batch_size={batch_size}, more_batch_dim={more_batch_dim}"

        # Create EnsembleFeedForwardNN instance
        hidden_dims = [17, 17, 17]
        ensemble_mlp = EnsembleFeedForwardNN(
            in_dim,
            out_dim,
            hidden_dims,
            num_ensemble,
            reduction=None,
            last_layer_is_readout=last_layer_is_readout,
            **kwargs,
        )

        # Test with a sample input
        input_tensor = torch.randn(batch_size, in_dim)
        ensemble_output = ensemble_mlp(input_tensor)

        # Check for the output shape
        self.assertEqual(ensemble_output.shape, (num_ensemble, batch_size, out_dim), msg=msg)

    def test_ensemble_feedforwardnn(self):
        # more_batch_dim=0
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=0
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=0
        )

        # more_batch_dim=1
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=1
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=1
        )

        # more_batch_dim=7
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=1, more_batch_dim=7
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=1, batch_size=13, more_batch_dim=7
        )

        # Test `last_layer_is_readout`
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, last_layer_is_readout=True
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, last_layer_is_readout=True
        )
        self.check_ensemble_feedforwardnn(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, last_layer_is_readout=True
        )

        # Test `reduction`
        self.check_ensemble_feedforwardnn_mean(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, last_layer_is_readout=True
        )
        self.check_ensemble_feedforwardnn_mean(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, last_layer_is_readout=True
        )
        self.check_ensemble_feedforwardnn_mean(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, last_layer_is_readout=True
        )

        # Test `subset_in_dim`
        self.check_ensemble_feedforwardnn_simple(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, subset_in_dim=0.5
        )
        self.check_ensemble_feedforwardnn_simple(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, subset_in_dim=0.5
        )
        self.check_ensemble_feedforwardnn_simple(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, subset_in_dim=0.5
        )
        self.check_ensemble_feedforwardnn_simple(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, subset_in_dim=7
        )
        self.check_ensemble_feedforwardnn_simple(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, subset_in_dim=7
        )
        self.check_ensemble_feedforwardnn_simple(
            in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, subset_in_dim=7
        )
        with self.assertRaises(AssertionError):
            self.check_ensemble_feedforwardnn_simple(
                in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=0, subset_in_dim=1.5
            )
        with self.assertRaises(AssertionError):
            self.check_ensemble_feedforwardnn_simple(
                in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=1, subset_in_dim=39
            )
        with self.assertRaises(AssertionError):
            self.check_ensemble_feedforwardnn_simple(
                in_dim=11, out_dim=5, num_ensemble=3, batch_size=13, more_batch_dim=7, subset_in_dim=39
            )


if __name__ == "__main__":
    ut.main()
