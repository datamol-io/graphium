import abc
from typing import Union, Callable, List, Optional, Mapping
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor, IntTensor
from torch_sparse import SparseTensor

from graphium.nn.base_layers import get_activation, DropPath
from graphium.utils.decorators import classproperty
import torch_geometric as pyg


class BaseGraphStructure:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        layer_idx: Optional[int] = None,
        layer_depth: Optional[int] = None,
        droppath_rate: float = 0.0,
    ):
        r"""
        Abstract class used to standardize the implementation of Pyg layers
        in the current library. It will allow a network to seemlesly swap between
        different GNN layers by better understanding the expected inputs
        and outputs.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            layer_idx:
                The index of the current layer

            layer_depth:
                The total depth (number of layers) associated to this specific layer

            droppath_rate:
                stochastic depth drop rate, between 0 and 1, see https://arxiv.org/abs/1603.09382
        """

        super().__init__()

        # Basic attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation
        self.layer_idx = layer_idx
        self.layer_depth = layer_depth
        self.droppath_rate = droppath_rate
        self._max_num_nodes_per_graph = None
        self._max_num_edges_per_graph = None

    def _initialize_activation_dropout_norm(self):
        if not isinstance(self, nn.Module):
            raise TypeError(
                "This function requires the current object to be an `nn.Module`. Use multi-inheritance or the class `BaseGraphModule` instead"
            )

        # Build the layers
        self.activation_layer = get_activation(self.activation)
        self.dropout_layer = self._parse_dropout(self.dropout)

        self.norm_layer = self._parse_norm(self.normalization)
        self.droppath_layer = self._parse_droppath(self.droppath_rate)

    def _parse_dropout(self, dropout):
        if callable(dropout):
            return deepcopy(dropout)
        elif dropout > 0:
            return nn.Dropout(p=dropout)
        return

    def _parse_droppath(self, droppath_rate):
        if droppath_rate == 0:
            return

        droppath_rate = DropPath.get_stochastic_drop_rate(droppath_rate, self.layer_idx, self.layer_depth)
        return DropPath(drop_rate=droppath_rate)

    def _parse_norm(self, normalization, dim=None):
        if dim is None:
            dim = self.out_dim * self.out_dim_factor
        if normalization is None or normalization == "none":
            parsed_norm = None
        elif callable(normalization):
            parsed_norm = deepcopy(normalization)
        elif normalization == "batch_norm":
            parsed_norm = nn.BatchNorm1d(dim)
        elif normalization == "layer_norm":
            parsed_norm = nn.LayerNorm(dim)
        else:
            raise ValueError(
                f"Undefined normalization `{normalization}`, must be `None`, `Callable`, 'batch_norm', 'layer_norm', 'none'"
            )
        return parsed_norm

    def apply_norm_activation_dropout(
        self,
        feat: Tensor,
        normalization: bool = True,
        activation: bool = True,
        dropout: bool = True,
        droppath: bool = True,
        batch_idx: Optional[IntTensor] = None,
        batch_size: Optional[int] = None,
    ):
        r"""
        Apply the different normalization and the dropout to the
        output layer.

        Parameters:

            feat:
                Feature tensor, to be normalized

            batch_idx

            normalization:
                Whether to apply the normalization

            activation:
                Whether to apply the activation layer

            dropout:
                Whether to apply the dropout layer

            droppath:
                Whether to apply the DropPath layer

        Returns:

            feat:
                Normalized and dropped-out features

        """

        if normalization and (self.norm_layer is not None):
            feat = self.norm_layer(feat)

        if activation and (self.activation_layer is not None):
            feat = self.activation_layer(feat)

        if dropout and (self.dropout_layer is not None):
            feat = self.dropout_layer(feat)

        if droppath and (self.droppath_layer is not None):
            feat = self.droppath_layer(feat, batch_idx=batch_idx, batch_size=batch_size)

        return feat

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        supports output edges edges or not.

        Returns:

            bool:
                Whether the layer supports the use of edges
        """
        ...

    @property
    @abc.abstractmethod
    def layer_inputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_input_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Whether the layer uses input edges in the forward pass
        """
        ...

    @property
    @abc.abstractmethod
    def layer_outputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_output_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Whether the layer outputs edges in the forward pass
        """
        ...

    @property
    @abc.abstractmethod
    def out_dim_factor(self) -> int:
        r"""
        Abstract method.
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns:

            int:
                The factor that multiplies the output dimensions
        """
        ...

    @property
    def max_num_nodes_per_graph(self) -> Optional[int]:
        """
        Get the maximum number of nodes per graph. Useful for reshaping a compiled model (IPU)
        """
        return self._max_num_nodes_per_graph

    @max_num_nodes_per_graph.setter
    def max_num_nodes_per_graph(self, value: Optional[int]):
        """
        Set the maximum number of nodes per graph. Useful for reshaping a compiled model (IPU)
        """
        if value is not None:
            assert isinstance(value, int) and (
                value > 0
            ), f"Value should be a positive integer, provided f{value} of type {type(value)}"
        self._max_num_nodes_per_graph = value

    @property
    def max_num_edges_per_graph(self) -> Optional[int]:
        """
        Get the maximum number of nodes per graph. Useful for reshaping a compiled model (IPU)
        """
        return self._max_num_edges_per_graph

    @max_num_edges_per_graph.setter
    def max_num_edges_per_graph(self, value: Optional[int]):
        """
        Set the maximum number of nodes per graph. Useful for reshaping a compiled model (IPU)
        """
        if value is not None:
            assert isinstance(value, int) and (
                value > 0
            ), f"Value should be a positive integer, provided f{value} of type {type(value)}"
        self._max_num_edges_per_graph = value

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        f = self.out_dim_factor
        out_dim_f_print = "" if f == 1 else f" * {f}"
        return f"{self.__class__.__name__}({self.in_dim} -> {self.out_dim}{out_dim_f_print}, activation={self.activation})"


class BaseGraphModule(BaseGraphStructure, nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        layer_idx: Optional[int] = None,
        layer_depth: Optional[int] = None,
        droppath_rate: float = 0.0,
    ):
        r"""
        Abstract class used to standardize the implementation of Pyg layers
        in the current library. It will allow a network to seemlesly swap between
        different GNN layers by better understanding the expected inputs
        and outputs.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            layer_idx:
                The index of the current layer

            layer_depth:
                The total depth (number of layers) associated to this specific layer

            droppath_rate:
                stochastic depth drop rate, between 0 and 1, see https://arxiv.org/abs/1603.09382
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
            layer_idx=layer_idx,
            layer_depth=layer_depth,
            droppath_rate=droppath_rate,
        )

        self._initialize_activation_dropout_norm()


def check_intpus_allow_int(obj, edge_index, size):
    """
    Overwrite the __check_input__ to allow for int32 and int16
    TODO: Remove when PyG and pytorch supports int32.
    """
    the_size: List[Optional[int]] = [None, None]

    if isinstance(edge_index, Tensor):
        # These 3 lines are different. They check for more int types and avoid overflow
        assert edge_index.dtype in (torch.long, torch.int64, torch.int32, torch.int16)
        # assert edge_index.min() >= 0
        # assert edge_index.max() < torch.iinfo(edge_index.dtype).max

        assert edge_index.dim() == 2
        assert edge_index.size(0) == 2
        if size is not None:
            the_size[0] = size[0]
            the_size[1] = size[1]
        return the_size

    elif isinstance(edge_index, SparseTensor):
        if obj.flow == "target_to_source":
            raise ValueError(
                (
                    'Flow direction "target_to_source" is invalid for '
                    "message propagation via `torch_sparse.SparseTensor`. If "
                    "you really want to make use of a reverse message "
                    "passing flow, pass in the transposed sparse tensor to "
                    "the message passing module, e.g., `adj_t.t()`."
                )
            )
        the_size[0] = edge_index.sparse_size(1)
        the_size[1] = edge_index.sparse_size(0)
        return the_size

    raise ValueError(
        (
            "`MessagePassing.propagate` only supports `torch.LongTensor` of "
            "shape `[2, num_messages]` or `torch_sparse.SparseTensor` for "
            "argument `edge_index`."
        )
    )
