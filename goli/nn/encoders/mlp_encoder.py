import torch
import torch.nn as nn
from typing import Union, Callable, List, Dict, Any, Optional
from torch_geometric.data import Batch

from goli.nn.base_layers import MLP, get_norm
from goli.nn.encoders.base_encoder import BaseEncoder


class MLPEncoder(BaseEncoder):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: str,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        activation: Union[str, Callable] = "relu",
        dropout=0.0,
        normalization="none",
        first_normalization="none",
        use_input_keys_prefix: bool = True,
    ):
        r"""
        Configurable kernel-based Positional Encoding node encoder.

        Parameters:
            input_keys: List of input keys to use from pyg batch graph
            output_keys: List of output keys to add to the pyg batch graph
            in_dim : input dimension of the mlp encoder
            hidden_dim : hidden dimension of the mlp encoder
            out_dim : output dimension of the mlp encoder
            num_layers : number of layers of the mlp encoder
            activation : activation function to use
            dropout : dropout to use
            normalization : normalization to use
            first_normalization : normalization to use before the first layer
            use_input_keys_prefix: Whether to use the `key_prefix` argument            
        """
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            first_normalization=first_normalization,
            use_input_keys_prefix=use_input_keys_prefix,
        )

        # Check the output_keys
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.normalization = get_norm(normalization)

        # Initialize the MLP
        self.pe_encoder = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            layers=num_layers,
            dropout=dropout,
            activation=activation,
            last_activation=activation,
            first_normalization=self.first_normalization,
            normalization=normalization,
            last_normalization=normalization,
            last_dropout=dropout,
        )

    def parse_input_keys(self, 
                         input_keys: List[str],
                         ) -> List[str]:
        r"""
        Parse the `input_keys`.
        Parameters:
            input_keys: List of input keys to use from pyg batch graph
        Returns:
            parsed input_keys
        """
        return input_keys

    def parse_output_keys(self, 
                          output_keys: List[str],
                          ) -> List[str]:
        r"""
        Parse the `output_keys`.
        Parameters:
            output_keys: List of output keys to add to the pyg batch graph
        Returns:
            parsed output_keys
        """
        assert len(output_keys) == len(
            self.input_keys
        ), f"The number of input keys {len(self.input_keys)} and output keys {len(output_keys)} must be the same for the class {self.__class__.__name__}"
        for in_key, out_key in zip(self.input_keys, output_keys):
            if in_key.startswith("edge_") or out_key.startswith("edge_"):
                assert out_key.startswith("edge_") and in_key.startswith(
                    "edge_"
                ), f"The output key {out_key} and input key {in_key} must match the 'edge_' prefix for the class {self.__class__.__name__}"
            if in_key.startswith("graph_") or out_key.startswith("graph_"):
                assert out_key.startswith("graph_") and in_key.startswith(
                    "graph_"
                ), f"The output key {out_key} and input key {in_key} must match the 'graph_' prefix for the class {self.__class__.__name__}"
        return output_keys

    def forward(self, 
                batch: Batch, 
                key_prefix: Optional[str] = None,
                ) -> Dict[str, torch.Tensor]:
        r"""
        forward function of the mlp encoder
        Parameters:
            batch: pyg batch graph
            key_prefix: Prefix to use for the input keys
        Returns:
            output: Dictionary of output embeddings with keys specified by input_keys
        """
        #TODO: maybe we should also use the output key here? @Dom 
        input_keys = self.parse_input_keys_with_prefix(key_prefix)

        # Run the MLP for each input key
        output = {}
        for key in input_keys:
            output[key] = self.pe_encoder(batch[key])  # (Num nodes) x dim_pe

        return output

    def make_mup_base_kwargs(self, 
                             divide_factor: float = 2.0, 
                             factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        Returns:
            base_kwargs: Dictionary of kwargs to use for the base model
        """
        base_kwargs = super().make_mup_base_kwargs(divide_factor=divide_factor, factor_in_dim=factor_in_dim)
        base_kwargs.update(
            dict(
                hidden_dim=round(self.hidden_dim / divide_factor),
                dropout=self.dropout,
                normalization=type(self.normalization).__name__,
            )
        )
        return base_kwargs
