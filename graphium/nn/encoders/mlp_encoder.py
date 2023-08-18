import torch
import torch.nn as nn
from typing import Union, Callable, List, Dict, Any, Optional
from torch_geometric.data import Batch

from graphium.nn.base_layers import MLP, get_norm
from graphium.nn.encoders.base_encoder import BaseEncoder


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
        Configurable kernel-based Positional Encoding node/edge-level encoder.

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
        self.normalization = normalization

        # Initialize the MLP
        self.pe_encoder = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dim,
            out_dim=out_dim,
            depth=num_layers,
            dropout=dropout,
            activation=activation,
            last_activation=activation,
            first_normalization=self.first_normalization,
            normalization=normalization,
            last_normalization=normalization,
            last_dropout=dropout,
        )

    def parse_input_keys(
        self,
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

    def parse_output_keys(
        self,
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
            if in_key.startswith("nodepair_") or out_key.startswith("nodepair_"):
                assert out_key.startswith("nodepair_") and in_key.startswith(
                    "nodepair_"
                ), f"The output key {out_key} and input key {in_key} must match the 'nodepair_' prefix for the class {self.__class__.__name__}"
            if in_key.startswith("graph_") or out_key.startswith("graph_"):
                assert out_key.startswith("graph_") and in_key.startswith(
                    "graph_"
                ), f"The output key {out_key} and input key {in_key} must match the 'graph_' prefix for the class {self.__class__.__name__}"
        return output_keys

    def forward(
        self,
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
        # TODO: maybe we should also use the output key here? @Dom
        # input_keys = self.parse_input_keys_with_prefix(key_prefix)

        # TODO: maybe it makes sense to combine MLPEncoder and CatMLPEncoder into one class with
        #   CatMLPEncoder being executed when the list input_keys contains more than one element.
        #   Currently, the input_keys list can only contain one element in MLPEncoder.

        # Run the MLP for each input key
        output = {}
        for input_key, output_key in zip(self.input_keys, self.output_keys):
            output[output_key] = self.pe_encoder(batch[input_key])  # (Num nodes/edges) x dim_pe

        return output

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
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
                normalization=self.normalization,
            )
        )
        return base_kwargs


class CatMLPEncoder(BaseEncoder):
    def __init__(
        self,
        input_keys: List[str],
        output_keys: str,
        in_dim: Union[int, List[int]],
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
        Configurable kernel-based Positional Encoding node/edge-level encoder.
        Concatenates the list of input (node or edge) features in the feature dimension

        Parameters:
            input_keys: List of input keys; inputs are concatenated in feat dimension and passed through mlp
            output_keys: List of output keys to add to the pyg batch graph
            in_dim : input dimension of the mlp encoder; sum of input dimensions of inputs
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

        if type(in_dim) is list:
            in_dim = sum(in_dim)

        # Check the output_keys
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.normalization = normalization

        # Initialize the MLP
        self.pe_encoder = MLP(
            in_dim=in_dim,
            hidden_dims=hidden_dim,
            out_dim=out_dim,
            depth=num_layers,
            dropout=dropout,
            activation=activation,
            last_activation=activation,
            first_normalization=self.first_normalization,
            normalization=normalization,
            last_normalization=normalization,
            last_dropout=dropout,
        )

    def parse_input_keys(
        self,
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

    def parse_output_keys(
        self,
        output_keys: List[str],
    ) -> List[str]:
        r"""
        Parse the `output_keys`.
        Parameters:
            output_keys: List of output keys to add to the pyg batch graph
        Returns:
            parsed output_keys
        """

        return output_keys

    def forward(
        self,
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
        # TODO: maybe we should also use the output key here? @Dom
        # input_keys = self.parse_input_keys_with_prefix(key_prefix)

        # Concatenate selected pes
        input = torch.cat(
            [batch[input_key] for input_key in self.input_keys], dim=-1
        )  # [num_nodes/num_edges, sum(in_dims)]

        output = {}
        for output_key in self.output_keys:
            output[output_key] = self.pe_encoder(input)

        return output

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
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
                normalization=self.normalization,
            )
        )
        return base_kwargs
