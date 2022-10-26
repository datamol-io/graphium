import abc
from typing import Union, Callable, List, Optional, Mapping
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor

from goli.nn.base_layers import get_activation
from goli.utils.decorators import classproperty

try:
    import dgl
except:
    dgl = None
try:
    import torch_geometric as pyg
except:
    pyg = None




class BaseGraphStructure:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
    ):
        r"""
        Abstract class used to standardize the implementation of DGL layers
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
        """

        super().__init__()

        # Basic attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.normalization = normalization
        self.dropout = dropout
        self.activation = activation

    def _initialize_activation_dropout_norm(self):

        if not isinstance(self, nn.Module):
            raise TypeError(
                "This function requires the current object to be an `nn.Module`. Use multi-inheritance or the class `BaseGraphModule` instead"
            )

        # Build the layers
        self.activation_layer = get_activation(self.activation)
        self.dropout_layer = self._parse_dropout(self.dropout)

        self.norm_layer = self._parse_norm(self.normalization)

    def _parse_dropout(self, dropout):
        if callable(dropout):
            return deepcopy(dropout)
        elif dropout > 0:
            return nn.Dropout(p=dropout)
        return

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
        h: Tensor,
        normalization: bool = True,
        activation: bool = True,
        dropout: bool = True,
    ):
        r"""
        Apply the different normalization and the dropout to the
        output layer.

        Parameters:

            h:
                Feature tensor, to be normalized

            normalization:
                Whether to apply the normalization

            activation:
                Whether to apply the activation layer

            dropout:
                Whether to apply the dropout layer

        Returns:

            h:
                Normalized and dropped-out features

        """

        if normalization and (self.norm_layer is not None):
            h = self.norm_layer(h)

        if activation and (self.activation_layer is not None):
            h = self.activation_layer(h)

        if dropout and (self.dropout_layer is not None):
            h = self.dropout_layer(h)

        return h

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
    ):
        r"""
        Abstract class used to standardize the implementation of DGL layers
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
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            normalization=normalization,
            dropout=dropout,
            activation=activation,
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


def get_node_feats(
    g: Union["dgl.DGLGraph", "pyg.data.Data", "pyg.data.Batch", Mapping], key: str = "h"
) -> Tensor:
    """
    Get the node features of a graph `g`.

    Parameters:
        g: graph
        key: key associated to the node features
    """
    if (dgl is not None) and isinstance(g, dgl.DGLGraph):
        return g.ndata.get(key, None)
    elif (pyg is not None) and isinstance(g, (pyg.data.Data, pyg.data.Batch)):
        return g.get(key, None)
    elif isinstance(g, Mapping):
        return g.get(key, None)
    else:
        raise TypeError(f"Unrecognized graph type {type(g)}")


def set_node_feats(
    g: Union["dgl.DGLGraph", "pyg.data.Data", "pyg.data.Batch", Mapping], node_feats: Tensor, key: str = "h"
) -> Tensor:
    """
    Set the node features of a graph `g`.

    Parameters:
        g: graph
        key: key associated to the node features
    """
    if (dgl is not None) and isinstance(g, dgl.DGLGraph):
        assert node_feats.shape[0] == g.num_nodes()
        g.ndata[key] = node_feats
    elif (pyg is not None) and isinstance(g, (pyg.data.Data, pyg.data.Batch)):
        assert node_feats.shape[0] == g.num_nodes
        g[key] = node_feats
    elif isinstance(g, Mapping):
        g[key] = node_feats
    else:
        raise TypeError(
            f"Unrecognized graph type {type(g)}. Make sure that pyg or dgl are installed if needed"
        )

    return g


def get_edge_feats(
    g: Union["dgl.DGLGraph", "pyg.data.Data", "pyg.data.Batch", Mapping], key: str = "h"
) -> Tensor:
    """
    Get the node features of a graph `g`.

    Parameters:
        g: graph
        key: key associated to the node features
    """
    if (dgl is not None) and isinstance(g, dgl.DGLGraph):
        return g.edata.get(key, None)
    elif (pyg is not None) and isinstance(g, (pyg.data.Data, pyg.data.Batch)):
        return g.get(key, None)
    elif isinstance(g, Mapping):
        return g.get(key, None)
    else:
        raise TypeError(f"Unrecognized graph type {type(g)}")


def set_edge_feats(
    g: Union["dgl.DGLGraph", "pyg.data.Data", "pyg.data.Batch", Mapping], edge_feats: Tensor, key: str = "h"
) -> Tensor:
    """
    Set the node features of a graph `g`.

    Parameters:
        g: graph
        key: key associated to the node features
    """
    if (dgl is not None) and isinstance(g, dgl.DGLGraph):
        if edge_feats is not None:
            assert edge_feats.shape[0] == g.num_edges()
            g.edata[key] = edge_feats
    elif (pyg is not None) and isinstance(g, (pyg.data.Data, pyg.data.Batch)):
        if edge_feats is not None:
            assert edge_feats.shape[0] == g.num_edges
            g[key] = edge_feats
    elif isinstance(g, Mapping):
        g[key] = edge_feats
    else:
        raise TypeError(
            f"Unrecognized graph type {type(g)}. Make sure that pyg or dgl are installed if needed"
        )

    return g
