import torch.nn as nn
import abc
from typing import List, Dict, Tuple

from goli.dgl.base_layers import get_activation


class BaseDGLLayer(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, activation="relu", dropout: float = 0.0, batch_norm: bool = False
    ):
        r"""
        Abstract class used to standardize the implementation of DGL layers
        in the current library. It will allow a network to seemlesly swap between
        different GNN layers by better understanding the expected inputs
        and outputs.

        Parameters
        ------------

        in_dim: int
            Input feature dimensions of the layer

        out_dim: int
            Output feature dimensions of the layer

        activation: str, Callable, Default="relu"
            activation function to use in the layer

        dropout: float, Default=0.
            The ratio of units to dropout. Must be between 0 and 1

        batch_norm: bool, Default=False
            Whether to use batch normalization
        """

        super().__init__()

        # Basic attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation

        # Build the layers
        self.activation_layer = get_activation(activation)

        self.dropout_layer = None
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)

        self.batch_norm_layer = None
        if batch_norm:
            self.batch_norm_layer = nn.BatchNorm1d(out_dim)

    def apply_norm_activation_dropout(
        self, h, batch_norm: bool = True, activation: bool = True, dropout: bool = True
    ):
        r"""
        Apply the different normalization and the dropout to the
        output layer.

        Parameters
        ------------

        h: torch.Tensor()
            Feature tensor, to be normalized

        batch_norm: bool, Default=True
            Whether to apply the batch_norm layer

        activation: bool, Default=True
            Whether to apply the activation layer

        dropout: bool, Default=True
            Whether to apply the dropout layer

        Returns
        ---------

        h: torch.Tensor()
            Normalized and dropped-out features

        """

        if batch_norm and (self.batch_norm_layer is not None):
            h = self.batch_norm_layer(h)

        if activation and (self.activation_layer is not None):
            h = self.activation_layer(h)

        if dropout and (self.dropout_layer is not None):
            h = self.dropout_layer(h)

        return h

    @staticmethod
    @abc.abstractmethod
    def layer_supports_edges() -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        supports edges or not.

        Returns
        ---------

        supports_edges: bool
            Whether the layer supports the use of edges
        """
        ...

    @abc.abstractmethod
    def layer_uses_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns
        ---------

        uses_edges: bool
            Whether the layer uses edges
        """
        ...

    @abc.abstractmethod
    def get_out_dim_factor(self) -> int:
        r"""
        Abstract method.
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns
        ---------

        dim_factor: int
            The factor that multiplies the dimensions
        """
        ...

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        return f"{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, activation={self.activation})"
