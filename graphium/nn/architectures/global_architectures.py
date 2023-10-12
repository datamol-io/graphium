from typing import Iterable, List, Dict, Literal, Tuple, Union, Callable, Any, Optional, Type
from torch_geometric.data import Batch
from graphium.ipu.to_dense_batch import to_dense_batch
from loguru import logger

# Misc imports
import inspect
from copy import deepcopy
from collections import OrderedDict

# Torch imports
from torch import Tensor, nn
import torch
from torch_geometric.data import Data

# graphium imports
from graphium.data.utils import get_keys
from graphium.nn.base_layers import FCLayer, get_activation, get_norm
from graphium.nn.architectures.encoder_manager import EncoderManager
from graphium.nn.pyg_layers import VirtualNodePyg, parse_pooling_layer_pyg
from graphium.nn.base_graph_layer import BaseGraphModule, BaseGraphStructure
from graphium.nn.residual_connections import (
    ResidualConnectionBase,
    ResidualConnectionWeighted,
    ResidualConnectionRandom,
)
from graphium.nn.utils import MupMixin
from graphium.ipu.ipu_utils import import_poptorch, is_running_on_ipu

poptorch = import_poptorch(raise_error=False)

import collections


class FeedForwardNN(nn.Module, MupMixin):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Union[List[int], int],
        depth: Optional[int] = None,
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        last_dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        first_normalization: Union[str, Callable] = "none",
        last_normalization: Union[str, Callable] = "none",
        residual_type: str = "none",
        residual_skip_steps: int = 1,
        name: str = "LNN",
        layer_type: Union[str, nn.Module] = "fc",
        layer_kwargs: Optional[Dict] = None,
        last_layer_is_readout: bool = False,
    ):
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            hidden_dims:
                Either an integer specifying all the hidden dimensions,
                or a list of dimensions in the hidden layers.
                Be careful, the "simple" residual type only supports
                hidden dimensions of the same value.

            depth:
                If `hidden_dims` is an integer, `depth` is 1 + the number of
                hidden layers to use.
                If `hidden_dims` is a list, then
                `depth` must be `None` or equal to `len(hidden_dims) + 1`

            activation:
                activation function to use in the hidden layers.

            last_activation:
                activation function to use in the last layer.

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            last_dropout:
                The ratio of units to dropout for the last_layer. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            first_normalization:
                Whether to use batch normalization **before** the first layer

            last_normalization:
                Whether to use batch normalization in the last layer

            residual_type:
                - "none": No residual connection
                - "simple": Residual connection similar to the ResNet architecture.
                  See class `ResidualConnectionSimple`
                - "weighted": Residual connection similar to the Resnet architecture,
                  but with weights applied before the summation. See class `ResidualConnectionWeighted`
                - "concat": Residual connection where the residual is concatenated instead
                  of being added.
                - "densenet": Residual connection where the residual of all previous layers
                  are concatenated. This leads to a strong increase in the number of parameters
                  if there are multiple hidden layers.

            residual_skip_steps:
                The number of steps to skip between each residual connection.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.

            name:
                Name attributed to the current network, for display and printing
                purposes.

            layer_type:
                The type of layers to use in the network.
                Either "fc" as the `FCLayer`, or a class representing the `nn.Module`
                to use.

            layer_kwargs:
                The arguments to be used in the initialization of the layer provided by `layer_type`

            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup

        """

        super().__init__()

        # Set the class attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * (depth - 1)
        else:
            self.hidden_dims = list(hidden_dims)
            assert (depth is None) or (
                depth == len(self.hidden_dims) + 1
            ), "Mismatch between the provided network depth from `hidden_dims` and `depth`"
        self.depth = len(self.hidden_dims) + 1
        self.activation = get_activation(activation)
        self.last_activation = get_activation(last_activation)
        self.dropout = dropout
        self.last_dropout = last_dropout
        self.normalization = normalization
        self.first_normalization = get_norm(first_normalization, dim=in_dim)
        self.last_normalization = last_normalization
        self.residual_type = None if residual_type is None else residual_type.lower()
        self.residual_skip_steps = residual_skip_steps
        self.layer_kwargs = layer_kwargs if layer_kwargs is not None else {}
        self.name = name
        self.last_layer_is_readout = last_layer_is_readout
        self._readout_cache = None

        # Parse the layer and residuals
        from graphium.utils.spaces import LAYERS_DICT, RESIDUALS_DICT

        self.layer_class, self.layer_name = self._parse_class_from_dict(layer_type, LAYERS_DICT)
        self.residual_class, self.residual_name = self._parse_class_from_dict(residual_type, RESIDUALS_DICT)

        self.full_dims = [self.in_dim] + self.hidden_dims + [self.out_dim]
        self._create_layers()
        self._check_bad_arguments()

    def _check_bad_arguments(self):
        r"""
        Raise comprehensive errors if the arguments seem wrong
        """
        if (self.residual_type == "simple") and not (self.hidden_dims[:-1] == self.hidden_dims[1:]):
            raise ValueError(
                f"When using the residual_type={self.residual_type}"
                + f", all elements in the hidden_dims must be equal. Provided:{self.hidden_dims}"
            )

    def _parse_class_from_dict(
        self, name_or_class: Union[type, str], class_dict: Dict[str, type]
    ) -> Tuple[type, str]:
        r"""
        Register the hyperparameters for tracking by Pytorch-lightning
        """
        if isinstance(name_or_class, str):
            obj_name = name_or_class.lower()
            obj_class = class_dict[obj_name]
        elif callable(name_or_class):
            obj_name = str(name_or_class)
            obj_class = name_or_class
        else:
            raise TypeError(f"`name_or_class` must be str or callable, provided: {type(name_or_class)}")

        return obj_class, obj_name

    def _create_residual_connection(self, out_dims: List[int]) -> Tuple[ResidualConnectionBase, List[int]]:
        r"""
        Create the residual connection classes.
        The out_dims is only used if the residual classes requires weights
        """
        if self.residual_class == ResidualConnectionWeighted:
            # if self.residual_class.has_weights:
            residual_layer = self.residual_class(
                skip_steps=self.residual_skip_steps,
                out_dims=out_dims,
                dropout=self.dropout,
                activation=self.activation,
                normalization=self.normalization,
                bias=False,
            )
        elif self.residual_class == ResidualConnectionRandom:
            residual_layer = self.residual_class(
                out_dims=out_dims,
                skip_steps=self.residual_skip_steps,
            )
        else:
            residual_layer = self.residual_class(skip_steps=self.residual_skip_steps)

        residual_out_dims = residual_layer.get_true_out_dims(self.full_dims[1:])

        return residual_layer, residual_out_dims

    def _create_layers(self):
        r"""
        Create all the necessary layers for the network.
        It's a bit complicated to explain what's going on in this function,
        but it must manage the varying features sizes caused by:

        - The presence of different types of residual connections
        """

        self.residual_layer, residual_out_dims = self._create_residual_connection(out_dims=self.full_dims[1:])

        # Create a ModuleList of the GNN layers
        self.layers = nn.ModuleList()
        this_in_dim = self.full_dims[0]
        this_activation = self.activation
        this_norm = self.normalization
        this_dropout = self.dropout

        for ii in range(self.depth):
            this_out_dim = self.full_dims[ii + 1]
            other_kwargs = {}
            sig = inspect.signature(self.layer_class)
            key_args = [p.name for p in sig.parameters.values()]
            if ii == self.depth - 1:
                this_activation = self.last_activation
                this_norm = self.last_normalization
                this_dropout = self.last_dropout
                if self.last_layer_is_readout and ("is_readout_layer" in key_args):
                    other_kwargs["is_readout_layer"] = self.last_layer_is_readout

            # Create the layer
            self.layers.append(
                self.layer_class(
                    in_dim=this_in_dim,
                    out_dim=this_out_dim,
                    activation=this_activation,
                    dropout=this_dropout,
                    normalization=this_norm,
                    **self.layer_kwargs,
                    **other_kwargs,
                )
            )

            if ii < len(residual_out_dims):
                this_in_dim = residual_out_dims[ii]

    @property
    def cache_readouts(self) -> bool:
        """Whether the readout cache is enabled"""
        return isinstance(self._readout_cache, dict)

    def _enable_readout_cache(self):
        """
        Enable the readout cache.
        Due to the usage of a dict, it only saves readouts for a single batch at a time
        """
        if not self.cache_readouts:
            self._readout_cache = {}

    def _disable_readout_cache(self):
        """Disable the readout cache"""
        self._readout_cache = None

    def drop_layers(self, depth: int) -> None:
        r"""
        Remove the last layers of the model part.
        """

        assert depth >= 0
        assert depth <= len(self.layers)

        if depth > 0:
            self.layers = self.layers[:-depth]

    def add_layers(self, layers: int) -> None:
        r"""
        Add layers to the end of the model.
        """
        assert isinstance(layers, nn.ModuleList)
        assert len(layers) > 0
        if len(self.layers) > 0:
            assert layers[0].in_dim == self.layers[-1].out_dim

        self.layers.extend(layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the neural network on the input features.

        Parameters:

            h: `torch.Tensor[..., Din]`:
                Input feature tensor, before the network.
                `Din` is the number of input features

        Returns:

            `torch.Tensor[..., Dout]`:
                Output feature tensor, after the network.
                `Dout` is the number of output features

        """
        feat_prev = None

        # Apply a normalization before the first layer
        if self.first_normalization is not None:
            h = self.first_normalization(h)

        # Apply all neural network layers
        for ii, layer in enumerate(self.layers):
            h = layer.forward(h)
            if ii < len(self.layers) - 1:
                h, feat_prev = self.residual_layer.forward(h, feat_prev, step_idx=ii)

                if self.cache_readouts:
                    self._readout_cache[ii] = h

        return h

    def get_init_kwargs(self) -> Dict[str, Any]:
        """
        Get a dictionary that can be used to instanciate a new object with identical parameters.
        """
        return deepcopy(
            dict(
                in_dim=self.in_dim,
                out_dim=self.out_dim,
                hidden_dims=self.hidden_dims,
                depth=None,
                activation=self.activation,
                last_activation=self.last_activation,
                dropout=self.dropout,
                last_dropout=self.last_dropout,
                normalization=self.normalization,
                first_normalization=self.first_normalization,
                last_normalization=self.last_normalization,
                residual_type=self.residual_type,
                residual_skip_steps=self.residual_skip_steps,
                name=self.name,
                layer_type=self.layer_class,
                layer_kwargs=self.layer_kwargs,
                last_layer_is_readout=self.last_layer_is_readout,
            )
        )

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension
        """
        kwargs = self.get_init_kwargs()
        kwargs["hidden_dims"] = [round(dim / divide_factor) for dim in kwargs["hidden_dims"]]
        if factor_in_dim:
            kwargs["in_dim"] = round(kwargs["in_dim"] / divide_factor)
        if not self.last_layer_is_readout:
            kwargs["out_dim"] = round(kwargs["out_dim"] / divide_factor)
        return kwargs

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        class_str = f"{self.name}(depth={self.depth}, {self.residual_layer})\n    "
        layer_str = f"[{self.layer_class.__name__}[{' -> '.join(map(str, self.full_dims))}]"

        return class_str + layer_str


class FeedForwardGraph(FeedForwardNN):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Union[List[int], int],
        layer_type: Union[str, nn.Module],
        depth: Optional[int] = None,
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        last_dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        first_normalization: Union[str, Callable] = "none",
        last_normalization: Union[str, Callable] = "none",
        residual_type: str = "none",
        residual_skip_steps: int = 1,
        in_dim_edges: int = 0,
        hidden_dims_edges: List[int] = [],
        name: str = "GNN",
        layer_kwargs: Optional[Dict] = None,
        virtual_node: str = "none",
        use_virtual_edges: bool = False,
        last_layer_is_readout: bool = False,
    ):
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different graph neural networks
        layers. Any layer must inherit from `graphium.nn.base_graph_layer.BaseGraphStructure`
        or `graphium.nn.base_graph_layer.BaseGraphLayer`.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            hidden_dims:
                List of dimensions in the hidden layers.
                Be careful, the "simple" residual type only supports
                hidden dimensions of the same value.

            layer_type:
                Type of layer to use. Can be a string or nn.Module.

            depth:
                If `hidden_dims` is an integer, `depth` is 1 + the number of
                hidden layers to use. If `hidden_dims` is a `list`, `depth` must
                be `None`.

            activation:
                activation function to use in the hidden layers.

            last_activation:
                activation function to use in the last layer.

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            last_dropout:
                The ratio of units to dropout for the last layer. Must be between 0 and 1

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            first_normalization:
                Whether to use batch normalization **before** the first layer

            last_normalization:
                Whether to use batch normalization in the last layer

            residual_type:
                - "none": No residual connection
                - "simple": Residual connection similar to the ResNet architecture.
                  See class `ResidualConnectionSimple`
                - "weighted": Residual connection similar to the Resnet architecture,
                  but with weights applied before the summation. See class `ResidualConnectionWeighted`
                - "concat": Residual connection where the residual is concatenated instead
                  of being added.
                - "densenet": Residual connection where the residual of all previous layers
                  are concatenated. This leads to a strong increase in the number of parameters
                  if there are multiple hidden layers.

            residual_skip_steps:
                The number of steps to skip between each residual connection.
                If `1`, all the layers are connected. If `2`, half of the
                layers are connected.

            in_dim_edges:
                Input edge-feature dimensions of the network. Keep at 0 if not using
                edge features, or if the layer doesn't support edges.

            hidden_dims_edges:
                Hidden dimensions for the edges. Most models don't support it, so it
                should only be used for those that do, i.e. `GatedGCNLayer`

            name:
                Name attributed to the current network, for display and printing
                purposes.

            layer_type:
                The type of layers to use in the network.
                A class that inherits from `graphium.nn.base_graph_layer.BaseGraphStructure`,
                or one of the following strings

                - "pyg:gin": GINConvPyg
                - "pyg:gine": GINEConvPyg
                - "pyg:gated-gcn": GatedGCNPyg
                - "pyg:pna-msgpass": PNAMessagePassingPyg

            layer_kwargs:
                The arguments to be used in the initialization of the layer provided by `layer_type`

            virtual_node:
                A string associated to the type of virtual node to use,
                either `None`, "none", "mean", "sum", "max", "logsum".
                See `graphium.nn.pooling_pyg.VirtualNode`.

                The virtual node will not use any residual connection if `residual_type`
                is "none". Otherwise, it will use a simple ResNet like residual
                connection.

            use_virtual_edges:
                A bool flag used to select if the virtual node should use the edges or not

            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup

        """

        # Initialize the additional attributes
        self.in_dim_edges = in_dim_edges
        if isinstance(hidden_dims_edges, int):
            self.hidden_dims_edges = [hidden_dims_edges] * (depth - 1)
        elif len(hidden_dims_edges) == 0:
            self.hidden_dims_edges = []
        else:
            self.hidden_dims_edges = list(hidden_dims_edges)
            assert depth is None
        self.full_dims_edges = None
        if len(self.hidden_dims_edges) > 0:
            self.full_dims_edges = [self.in_dim_edges] + self.hidden_dims_edges + [self.hidden_dims_edges[-1]]

        self.virtual_node = virtual_node.lower() if virtual_node is not None else "none"

        self.use_virtual_edges = use_virtual_edges
        self.virtual_node_class = self._parse_virtual_node_class()

        # Initialize the parent `FeedForwardNN`
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            activation=activation,
            last_activation=last_activation,
            normalization=normalization,
            first_normalization=first_normalization,
            last_normalization=last_normalization,
            residual_type=residual_type,
            residual_skip_steps=residual_skip_steps,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            last_dropout=last_dropout,
            layer_kwargs=layer_kwargs,
            last_layer_is_readout=last_layer_is_readout,
        )

        self.first_normalization_edges = get_norm(first_normalization, dim=in_dim_edges)

    def _check_bad_arguments(self):
        r"""
        Raise comprehensive errors if the arguments seem wrong
        """
        super()._check_bad_arguments()
        if (
            (self.in_dim_edges > 0) or (self.full_dims_edges is not None)
        ) and not self.layer_class.layer_supports_edges:
            raise ValueError(f"Cannot use edge features with class `{self.layer_class}`")

    def _create_layers(self):
        r"""
        Create all the necessary layers for the network.
        It's a bit complicated to explain what's going on in this function,
        but it must manage the varying features sizes caused by:

        - The presence of different types of residual connections
        - The presence or absence of edges
        - The output dimensions varying for different networks i.e. `GatLayer` outputs different feature sizes according to the number of heads
        - The presence or absence of virtual nodes
        - The different possible pooling, and the concatenation of multiple pooling together.
        """

        residual_layer_temp, residual_out_dims = self._create_residual_connection(out_dims=self.full_dims[1:])

        # Create a ModuleList of the GNN layers
        self.layers = nn.ModuleList()
        self.virtual_node_layers = nn.ModuleList()
        this_in_dim = self.full_dims[0]
        this_activation = self.activation
        this_norm = self.normalization
        this_dropout = self.dropout

        # Find the appropriate edge dimensions, depending if edges are used,
        # And if the residual is required for the edges
        this_in_dim_edges, this_out_dim_edges = None, None
        if self.full_dims_edges is not None:
            this_in_dim_edges, this_out_dim_edges = self.full_dims_edges[0:2]
            residual_out_dims_edges = residual_layer_temp.get_true_out_dims(self.full_dims_edges[1:])
        elif self.in_dim_edges > 0:
            this_in_dim_edges = self.in_dim_edges
        layer_out_dims_edges = []

        # Create all the layers in a loop
        for ii in range(self.depth):
            this_out_dim = self.full_dims[ii + 1]

            # Find the edge key-word arguments depending on the layer type and residual connection
            this_edge_kwargs = {}
            if self.layer_class.layer_supports_edges and self.in_dim_edges > 0:
                this_edge_kwargs["in_dim_edges"] = this_in_dim_edges
                if "out_dim_edges" in inspect.signature(self.layer_class.__init__).parameters.keys():
                    if self.full_dims_edges is not None:
                        this_out_dim_edges = self.full_dims_edges[ii + 1]
                        this_edge_kwargs["out_dim_edges"] = this_out_dim_edges
                    else:
                        this_out_dim_edges = self.layer_kwargs.get("out_dim_edges")
                    layer_out_dims_edges.append(this_out_dim_edges)

            # Create the GNN layer
            self.layers.append(
                self.layer_class(
                    in_dim=this_in_dim,
                    out_dim=this_out_dim,
                    activation=this_activation,
                    dropout=this_dropout,
                    normalization=this_norm,
                    layer_idx=ii,
                    layer_depth=self.depth,
                    **self.layer_kwargs,
                    **this_edge_kwargs,
                )
            )

            # Create the Virtual Node layer, except at the last layer
            if ii < len(residual_out_dims):
                self.virtual_node_layers.append(
                    self.virtual_node_class(
                        in_dim=this_out_dim * self.layers[-1].out_dim_factor,
                        out_dim=this_out_dim * self.layers[-1].out_dim_factor,
                        in_dim_edges=this_out_dim_edges,
                        out_dim_edges=this_out_dim_edges,
                        activation=this_activation,
                        dropout=this_dropout,
                        normalization=this_norm,
                        bias=True,
                        vn_type=self.virtual_node,
                        residual=self.residual_type is not None,
                        use_edges=self.use_virtual_edges,
                    )
                )

            # Get the true input dimension of the next layer,
            # by factoring both the residual connection and GNN layer type
            if ii < len(residual_out_dims):
                this_in_dim = residual_out_dims[ii] * self.layers[ii - 1].out_dim_factor
                if self.full_dims_edges is not None:
                    this_in_dim_edges = residual_out_dims_edges[ii] * self.layers[ii - 1].out_dim_factor

        layer_out_dims = [layer.out_dim_factor * layer.out_dim for layer in self.layers]

        # Initialize residual and pooling layers
        self.residual_layer, _ = self._create_residual_connection(out_dims=layer_out_dims)
        if len(layer_out_dims_edges) > 0:
            self.residual_edges_layer, _ = self._create_residual_connection(out_dims=layer_out_dims_edges)
        else:
            self.residual_edges_layer = None

    def _graph_layer_forward(
        self,
        layer: BaseGraphModule,
        g: Batch,
        feat: Tensor,
        edge_feat: Optional[Tensor],
        feat_prev: Optional[Tensor],
        edge_feat_prev: Optional[Tensor],
        step_idx: int,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different PyG-based graph neural networks
        layers. Any layer must inherit from `graphium.nn.base_graph_layer.BaseGraphStructure`
        or `graphium.nn.base_graph_layer.BaseGraphLayer`.

        Apply the *i-th* PyG graph layer, where *i* is the index given by `step_idx`.
        The layer is applied differently depending if there are edge features or not.

        Then, the residual is also applied on both the features and the edges (if applicable)

        Parameters:

            layer:
                The PyG layer used for the convolution

            g:
                graph on which the convolution is done

            feat (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            edge_feat (torch.Tensor[..., N, Ein]):
                Edge feature tensor, before convolution.
                `N` is the number of nodes, `Ein` is the input edge features

            feat_prev:
                Node feature of the previous residual connection, or `None`

            edge_feat_prev:
                Edge feature of the previous residual connection, or `None`

            step_idx:
                The current step idx in the forward loop

        Returns:

            feat (torch.Tensor[..., N, Dout]):
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            edge_feat:
                Edge feature tensor, after convolution and residual.
                `N` is the number of nodes, `Ein` is the input edge features

            feat_prev:
                Node feature tensor to be used at the next residual connection, or `None`

            edge_feat_prev:
                Edge feature tensor to be used at the next residual connection, or `None`

        """

        # Set node / edge features into the graph
        g["feat"] = feat
        g["edge_feat"] = edge_feat

        # Apply the GNN layer
        g = layer(g)

        # Get the node / edge features from the graph
        feat = g["feat"]
        edge_feat = g["edge_feat"]

        # Apply the residual layers on the features and edges (if applicable)
        if step_idx < len(self.layers) - 1:
            feat, feat_prev = self.residual_layer.forward(feat, feat_prev, step_idx=step_idx)
            if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
                edge_feat, edge_feat_prev = self.residual_edges_layer.forward(
                    edge_feat, edge_feat_prev, step_idx=step_idx
                )

        return feat, edge_feat, feat_prev, edge_feat_prev

    def _parse_virtual_node_class(self) -> type:
        return VirtualNodePyg

    def _virtual_node_forward(
        self,
        g: Data,
        feat: torch.Tensor,
        vn_feat: torch.Tensor,
        step_idx: int,
        edge_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the *i-th* virtual node layer, where *i* is the index given by `step_idx`.

        Parameters:

            g:
                graph on which the convolution is done

            feat (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            vn_feat (torch.Tensor[..., M, Din]):
                Graph feature of the previous virtual node, or `None`
                `M` is the number of graphs, `Din` is the input features
                It is added to the result after the MLP, as a residual connection

            step_idx:
                The current step idx in the forward loop

            edge_feat: torch.Tensor

        Returns:

            `feat = torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            `vn_feat = torch.Tensor[..., M, Dout]`:
                Graph feature tensor to be used at the next virtual node, or `None`
                `M` is the number of graphs, `Dout` is the output features

        """
        if step_idx < len(self.virtual_node_layers):
            feat, vn_feat, edge_feat = self.virtual_node_layers[step_idx].forward(
                g=g, feat=feat, vn_feat=vn_feat, edge_feat=edge_feat
            )

        return feat, vn_feat, edge_feat

    def forward(self, g: Batch) -> torch.Tensor:
        r"""
        Apply the full graph neural network on the input graph and node features.

        Parameters:

            g:
                pyg Batch graph on which the convolution is done with the keys:

                - `"feat"`: torch.Tensor[..., N, Din]
                  Node feature tensor, before convolution.
                  `N` is the number of nodes, `Din` is the input features

                - `"edge_feat"` (torch.Tensor[..., N, Ein]):
                  Edge feature tensor, before convolution.
                  `N` is the number of nodes, `Ein` is the input edge features


        Returns:

            `torch.Tensor[..., M, Dout]` or `torch.Tensor[..., N, Dout]`:
                Node or graph feature tensor, after the network.
                `N` is the number of nodes, `M` is the number of graphs,
                `Dout` is the output dimension ``self.out_dim``
                If the `self.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        # Initialize values of the residuals and virtual node
        feat_prev = None
        edge_feat_prev = None
        vn_feat = 0.0
        feat = g["feat"]
        edge_feat = g["edge_feat"]
        # Apply the normalization before the first network layers
        if self.first_normalization is not None:
            feat = self.first_normalization(feat)
        if (self.first_normalization_edges is not None) and (self.in_dim_edges > 0):
            edge_feat = self.first_normalization_edges(edge_feat)

        # Apply the forward loop of the layers, residuals and virtual nodes
        for ii, layer in enumerate(self.layers):
            feat, edge_feat, feat_prev, edge_feat_prev = self._graph_layer_forward(
                layer=layer,
                g=g,
                feat=feat,
                edge_feat=edge_feat,
                feat_prev=feat_prev,
                edge_feat_prev=edge_feat_prev,
                step_idx=ii,
            )
            feat, vn_feat, edge_feat = self._virtual_node_forward(
                g=g, feat=feat, edge_feat=edge_feat, vn_feat=vn_feat, step_idx=ii
            )

            if self.cache_readouts:
                self._readout_cache[ii] = feat

        g["feat"], g["edge_feat"] = feat, edge_feat
        return g

    def get_init_kwargs(self) -> Dict[str, Any]:
        """
        Get a dictionary that can be used to instanciate a new object with identical parameters.
        """
        kwargs = super().get_init_kwargs()
        new_kwargs = dict(
            in_dim_edges=self.in_dim_edges,
            hidden_dims_edges=self.hidden_dims_edges,
            virtual_node=self.virtual_node,
            use_virtual_edges=self.use_virtual_edges,
        )
        kwargs.update(new_kwargs)
        return deepcopy(kwargs)

    def make_mup_base_kwargs(
        self,
        divide_factor: float = 2.0,
        factor_in_dim: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension for the nodes

        Returns:
            kwargs: Dictionary of parameters to be used to instanciate the base model divided by the factor
        """
        kwargs = self.get_init_kwargs()
        kwargs["hidden_dims"] = [round(dim / divide_factor) for dim in kwargs["hidden_dims"]]
        kwargs["hidden_dims_edges"] = [round(dim / divide_factor) for dim in kwargs["hidden_dims_edges"]]
        if factor_in_dim:
            kwargs["in_dim"] = round(kwargs["in_dim"] / divide_factor)
            kwargs["in_dim_edges"] = round(kwargs["in_dim_edges"] / divide_factor)
        if not self.last_layer_is_readout:
            kwargs["out_dim"] = round(kwargs["out_dim"] / divide_factor)

        def _recursive_divide_dim(x: collections.abc.Mapping):
            for k, v in x.items():
                if isinstance(v, collections.abc.Mapping):
                    _recursive_divide_dim(v)
                elif k in ["in_dim", "out_dim", "in_dim_edges", "out_dim_edges"]:
                    x[k] = round(v / divide_factor)

        _recursive_divide_dim(kwargs["layer_kwargs"])

        return kwargs

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        class_str = f"{self.name}(depth={self.depth}, {self.residual_layer})\n    "
        layer_str = f"{self.layer_class.__name__}[{' -> '.join(map(str, self.full_dims))}]\n    "

        return class_str + layer_str


class FullGraphMultiTaskNetwork(nn.Module, MupMixin):
    def __init__(
        self,
        gnn_kwargs: Dict[str, Any],
        pre_nn_kwargs: Optional[Dict[str, Any]] = None,
        pre_nn_edges_kwargs: Optional[Dict[str, Any]] = None,
        pe_encoders_kwargs: Optional[Dict[str, Any]] = None,
        task_heads_kwargs: Optional[Dict[str, Any]] = None,
        graph_output_nn_kwargs: Optional[Dict[str, Any]] = None,
        accelerator_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_to_average: int = 1,
        last_layer_is_readout: bool = False,
        name: str = "FullGNN",
    ):
        r"""
        Class that allows to implement a full graph neural network architecture,
        including the pre-processing MLP and the post processing MLP.

        Parameters:

            gnn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                GNN network using the class `FeedForwardGraph`.
                It must respect the following criteria:

                - gnn_kwargs["in_dim"] must be equal to pre_nn_kwargs["out_dim"]
                - gnn_kwargs["out_dim"] must be equal to graph_output_nn_kwargs["in_dim"]

            pe_encoders_kwargs:
                key-word arguments to use for the initialization of all positional encoding encoders.
                See the class `EncoderManager` for more details.

            pre_nn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the node features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            pre_nn_edges_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the edge features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            task_heads_kwargs:
                This argument is a list of dictionaries containing the arguments for task heads. Each argument is used to
                initialize a task-specific MLP.

            graph_output_nn_kwargs:
                This argument is a list of dictionaries corresponding to the arguments for a FeedForwardNN.
                Each dict of arguments is used to initialize a shared MLP.

            accelerator_kwargs:
                key-word arguments specific to the accelerator being used,
                e.g. pipeline split points

            num_inference_to_average:
                Number of inferences to average at val/test time. This is used to avoid the noise introduced
                by positional encodings with sign-flips. In case no such encoding is given,
                this parameter is ignored.
                NOTE: The inference time will be slowed-down proportionaly to this parameter.

            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method https://github.com/microsoft/mup

            name:
                Name attributed to the current network, for display and printing
                purposes.
        """

        super().__init__()
        self.name = name
        self.num_inference_to_average = num_inference_to_average
        self.last_layer_is_readout = last_layer_is_readout
        self._concat_last_layers = None
        self.pre_nn, self.pre_nn_edges, self.task_heads = None, None, None
        self.pe_encoders_kwargs = deepcopy(pe_encoders_kwargs)
        self.graph_output_nn_kwargs = graph_output_nn_kwargs
        self.encoder_manager = EncoderManager(pe_encoders_kwargs)
        self.max_num_nodes_per_graph = None
        self.max_num_edges_per_graph = None
        self._cache_readouts = False

        # Initialize the pre-processing neural net for nodes (applied directly on node features)
        if pre_nn_kwargs is not None:
            name = pre_nn_kwargs.pop("name", "pre-NN")
            self.pre_nn = FeedForwardNN(**pre_nn_kwargs, name=name)
            next_in_dim = self.pre_nn.out_dim
            gnn_kwargs.setdefault("in_dim", next_in_dim)
            assert (
                next_in_dim == gnn_kwargs["in_dim"]
            ), f"Inconsistent dimensions between pre-NN output ({next_in_dim}) and GNN input ({gnn_kwargs['in_dim']})"

        # Initialize the pre-processing neural net for edges (applied directly on edge features)
        if pre_nn_edges_kwargs is not None:
            name = pre_nn_edges_kwargs.pop("name", "pre-NN-edges")
            self.pre_nn_edges = FeedForwardNN(**pre_nn_edges_kwargs, name=name)
            next_in_dim = self.pre_nn_edges.out_dim
            gnn_kwargs.setdefault("in_dim_edges", next_in_dim)
            assert (
                next_in_dim == gnn_kwargs["in_dim_edges"]
            ), f"Inconsistent dimensions between pre-NN-edges output ({next_in_dim}) and GNN input ({gnn_kwargs['in_dim_edges']})"

        # Initialize the graph neural net (applied after the pre_nn)
        name = gnn_kwargs.pop("name", "GNN")
        gnn_class = self._parse_feed_forward_gnn(gnn_kwargs)
        gnn_kwargs.setdefault(
            "last_layer_is_readout", self.last_layer_is_readout and (task_heads_kwargs is None)
        )
        self.gnn = gnn_class(**gnn_kwargs, name=name)
        next_in_dim = self.gnn.out_dim

        if task_heads_kwargs is not None:
            self.task_heads = TaskHeads(
                in_dim=self.out_dim,
                in_dim_edges=self.out_dim_edges,
                task_heads_kwargs=task_heads_kwargs,
                graph_output_nn_kwargs=graph_output_nn_kwargs,
            )
            self._task_heads_kwargs = task_heads_kwargs

        if accelerator_kwargs is not None:
            accelerator = accelerator_kwargs["_accelerator"]
            if accelerator == "ipu":
                self._apply_ipu_options(accelerator_kwargs)

        self._check_bad_arguments()

    @staticmethod
    def _parse_feed_forward_gnn(
        gnn_kwargs: Dict[str, Any],
    ):
        """
        Parse the key-word arguments to determine which `FeedForward` class to use.
        Parameters:
            gnn_kwargs: key-word arguments to use for the initialization of the GNN network.

        Returns:
            `FeedForwardPyg` class
        """

        return FeedForwardGraph

    def _check_bad_arguments(self):
        r"""
        Raise comprehensive errors if the arguments seem wrong
        """
        if self.pre_nn is not None:
            if self.pre_nn.out_dim != self.gnn.in_dim:
                raise ValueError(
                    f"`self.pre_nn.out_dim` must be equal to `self.gnn.in_dim`."
                    + 'Provided" {self.pre_nn.out_dim} and {self.gnn.in_dim}'
                )

        if self.task_heads is not None:
            edge_level_tasks = [
                task_name
                for task_name, head_kwargs in self._task_heads_kwargs.items()
                if head_kwargs["task_level"] == "edge"
            ]
            if len(edge_level_tasks) > 0 and (
                not self.gnn.layer_class.layer_supports_edges or self.out_dim_edges < 1
            ):
                raise ValueError(
                    f"Task heads have edge level tasks {', '.join(edge_level_tasks)}, but edge level tasks cannot be used with layer class `{self.gnn.layer_class}`"
                )
            graph_level_tasks = [
                task_name
                for task_name, head_kwargs in self._task_heads_kwargs.items()
                if head_kwargs["task_level"] == "graph"
            ]
            if len(graph_level_tasks) > 0 and self.graph_output_nn_kwargs["graph"]["pooling"] == ["none"]:
                raise ValueError(
                    f"Task heads have graph level tasks {', '.join(graph_level_tasks)}, but pooling is none."
                )

    def _apply_ipu_options(self, ipu_kwargs):
        gnn_layers_per_ipu = ipu_kwargs.get("gnn_layers_per_ipu")
        self._apply_ipu_pipeline_split(gnn_layers_per_ipu)

    def _apply_ipu_pipeline_split(self, gnn_layers_per_ipu):
        r"""
        Apply pipeline split from accelerator options if applicable
        """

        if gnn_layers_per_ipu is None:
            return

        if not isinstance(gnn_layers_per_ipu, collections.abc.Sequence):
            raise ValueError("gnn_layers_per_ipu must be a Sequence (e.g. a list)")

        valid_ipu_pipeline_lengths = [1, 2, 4, 8, 16]
        pipeline_length = len(gnn_layers_per_ipu)

        if pipeline_length not in valid_ipu_pipeline_lengths:
            raise ValueError(
                f"Length of gnn_layers_per_ipu must be one of {valid_ipu_pipeline_lengths}, "
                f"got {gnn_layers_per_ipu} of length {pipeline_length} instead"
            )

        model_depth = len(self.gnn.layers)

        if sum(gnn_layers_per_ipu) != model_depth:
            raise ValueError(
                f"The values in gnn_layers_per_ipu must add up to the depth of the model, "
                f"got {gnn_layers_per_ipu} with total {sum(gnn_layers_per_ipu)} vs model depth "
                f"of {model_depth}"
            )

        begin_block_layer_indices = [sum(gnn_layers_per_ipu[:i]) for i in range(1, pipeline_length)]

        for begin_block_layer_index, ipu_id in zip(begin_block_layer_indices, range(1, pipeline_length)):
            self.gnn.layers[begin_block_layer_index] = poptorch.BeginBlock(
                self.gnn.layers[begin_block_layer_index], ipu_id=ipu_id
            )

    def _enable_readout_cache(self, module_filter: Optional[Union[str, List[str]]]):
        """
        Enable a single-batch readout cache for (a subset of) the modules.
        This is used to extract hidden representations for fingerprinting.
        """

        self.create_module_map(level="module")

        for k in module_filter:
            if k not in self._module_map:
                raise ValueError(f"Module {k} not found in network, choose from {self._module_map.keys()}")

        if module_filter is None:
            module_filter = list(self._module_map.keys())
        if isinstance(module_filter, str):
            module_filter = [module_filter]

        for module_name, module in self._module_map.items():
            if module_name in module_filter:
                if not isinstance(module, FeedForwardNN):
                    raise RuntimeError(
                        f"Readout cache can only be enabled for FeedForwardNN subclasses, not {type(module)}"
                    )
                module._enable_readout_cache()

        self._cache_readouts = True

    def _disable_readout_cache(self):
        """Disable the readout cache"""
        self.create_module_map(level="module")
        for _, module in self._module_map.items():
            if isinstance(module, FeedForwardNN):
                module._disable_readout_cache()
        self._cache_readouts = False

    def create_module_map(self, level: Union[Literal["layers"], Literal["module"]] = "layers"):
        """
        Function to create mapping between each (sub)module name and corresponding nn.ModuleList() (if possible);
        Used for finetuning when (partially) loading or freezing specific modules of the pretrained model

        Args:
            level: Whether to map to the module object or the layers of the module object
        """
        self._module_map = OrderedDict()

        if self.encoder_manager is not None:
            self._module_map.update(
                {"pe_encoders": self.encoder_manager}
            )  # could be extended to submodules, e.g. pe_encoders/la_pos/linear_in/..., etc.; not necessary for current finetuning

        if self.pre_nn is not None:
            self._module_map.update({"pre_nn": self.pre_nn})

        if self.pre_nn_edges is not None:
            self._module_map.update({"pre_nn_edges": self.pre_nn_edges})

        # No need to check for NoneType as GNN module is not optional in FullGraphMultitaskNetwork
        self._module_map.update({"gnn": self.gnn})

        if self.task_heads is not None:
            self._module_map.update(
                {
                    "graph_output_nn-"
                    + output_level: self.task_heads.graph_output_nn[output_level].graph_output_nn
                    for output_level in self.task_heads.graph_output_nn.keys()
                }
            )

            self._module_map.update(
                {
                    "task_heads-" + task_head_name: self.task_heads.task_heads[task_head_name]
                    for task_head_name in self.task_heads.task_heads.keys()
                }
            )

            if level == "layers":
                for module_name, module in self._module_map.items():
                    if module_name != "pe_encoders":
                        self._module_map[module_name] = module.layers
        return self._module_map

    def forward(self, g: Batch) -> Tensor:
        r"""
        Apply the pre-processing neural network, the graph neural network,
        and the post-processing neural network on the graph features.

        Parameters:

            g:
                pyg Batch graph on which the convolution is done.
                Must contain the following elements:

                - Node key `"feat"`: `torch.Tensor[..., N, Din]`.
                  Input node feature tensor, before the network.
                  `N` is the number of nodes, `Din` is the input features dimension ``self.pre_nn.in_dim``

                - Edge key `"edge_feat"`: `torch.Tensor[..., N, Ein]` **Optional**.
                  The edge features to use. It will be ignored if the
                  model doesn't supporte edge features or if
                  `self.in_dim_edges==0`.

                - Other keys related to positional encodings `"pos_enc_feats_sign_flip"`,
                  `"pos_enc_feats_no_flip"`.

        Returns:

            `torch.Tensor[..., M, Dout]` or `torch.Tensor[..., N, Dout]`:
                Node or graph feature tensor, after the network.
                `N` is the number of nodes, `M` is the number of graphs,
                `Dout` is the output dimension ``self.graph_output_nn.out_dim``
                If the `self.gnn.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        # Apply the positional encoders
        g = self.encoder_manager(g)

        e = None

        # Run the pre-processing network on node features
        if self.pre_nn is not None:
            g["feat"] = self.pre_nn.forward(g["feat"])

        # Run the pre-processing network on edge features
        # If there are no edges, skip the forward and change the dimension of e
        if self.pre_nn_edges is not None:
            e = g["edge_feat"]
            if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                e = torch.zeros(
                    list(e.shape[:-1]) + [self.pre_nn_edges.out_dim], device=e.device, dtype=e.dtype
                )
            else:
                e = self.pre_nn_edges.forward(e)
            g["edge_feat"] = e

        # Run the graph neural network
        g = self.gnn.forward(g)

        if self.task_heads is not None:
            return self.task_heads.forward(g)

        return g

    def make_mup_base_kwargs(self, divide_factor: float = 2.0) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.

        Returns:
            Dictionary with the kwargs to create the base model.
        """
        kwargs = dict(
            gnn_kwargs=None,
            pre_nn_kwargs=None,
            pre_nn_edges_kwargs=None,
            pe_encoders_kwargs=None,
            num_inference_to_average=self.num_inference_to_average,
            last_layer_is_readout=self.last_layer_is_readout,
            name=self.name,
        )

        # For the pre-nn network, get the smaller dimensions.
        # For the input dim, only divide the features coming from the pe-encoders
        if self.pre_nn is not None:
            kwargs["pre_nn_kwargs"] = self.pre_nn.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=False
            )
            pe_enc_outdim = 0 if self.encoder_manager is None else self.pe_encoders_kwargs.get("out_dim", 0)
            pre_nn_indim = kwargs["pre_nn_kwargs"]["in_dim"] - pe_enc_outdim
            kwargs["pre_nn_kwargs"]["in_dim"] = round(pre_nn_indim + (pe_enc_outdim / divide_factor))

        # For the pre-nn on the edges, factor all dimensions, except the in_dim
        if self.pre_nn_edges is not None:
            kwargs["pre_nn_edges_kwargs"] = self.pre_nn_edges.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=False
            )
            pe_enc_edge_outdim = (
                0 if self.encoder_manager is None else self.pe_encoders_kwargs.get("edge_out_dim", 0)
            )
            pre_nn_edge_indim = kwargs["pre_nn_edges_kwargs"]["in_dim"] - pe_enc_edge_outdim
            kwargs["pre_nn_edges_kwargs"]["in_dim"] = round(
                pre_nn_edge_indim + (pe_enc_edge_outdim / divide_factor)
            )

        # For the pe-encoders, don't factor the in_dim and in_dim_edges
        if self.encoder_manager is not None:
            kwargs["pe_encoders_kwargs"] = self.encoder_manager.make_mup_base_kwargs(
                divide_factor=divide_factor
            )

        if self.task_heads is not None:
            task_heads_kwargs = self.task_heads.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=True
            )
            kwargs["task_heads_kwargs"] = task_heads_kwargs["task_heads_kwargs"]
            kwargs["graph_output_nn_kwargs"] = task_heads_kwargs["graph_output_nn_kwargs"]

        # For the gnn network, all the dimension are divided, except the input dims if pre-nn are missing
        if self.gnn is not None:
            factor_in_dim = self.pre_nn is not None
            kwargs["gnn_kwargs"] = self.gnn.make_mup_base_kwargs(
                divide_factor=divide_factor,
                factor_in_dim=factor_in_dim,
            )

        return kwargs

    def set_max_num_nodes_edges_per_graph(self, max_nodes: Optional[int], max_edges: Optional[int]) -> None:
        """
        Set the maximum number of nodes and edges for all gnn layers and encoder layers

        Parameters:
            max_nodes: Maximum number of nodes in the dataset.
                This will be useful for certain architecture, but ignored by others.

            max_edges: Maximum number of edges in the dataset.
                This will be useful for certain architecture, but ignored by others.
        """
        self.max_num_nodes_per_graph = max_nodes
        self.max_num_edges_per_graph = max_edges
        if (self.encoder_manager is not None) and (self.encoder_manager.pe_encoders is not None):
            for encoder in self.encoder_manager.pe_encoders.values():
                encoder.max_num_nodes_per_graph = max_nodes
                encoder.max_num_edges_per_graph = max_edges
        if self.gnn is not None:
            for layer in self.gnn.layers:
                if isinstance(layer, BaseGraphStructure):
                    layer.max_num_nodes_per_graph = max_nodes
                    layer.max_num_edges_per_graph = max_edges

        self.task_heads.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)

    def __repr__(self) -> str:
        r"""
        Controls how the class is printed
        """
        pre_nn_str, pre_nn_edges_str = "", ""
        if self.pre_nn is not None:
            pre_nn_str = self.pre_nn.__repr__() + "\n\n"
        if self.pre_nn_edges is not None:
            pre_nn_edges_str = self.pre_nn_edges.__repr__() + "\n\n"
        gnn_str = self.gnn.__repr__() + "\n\n"
        if self.task_heads is not None:
            task_str = self.task_heads.__repr__()
            task_str = "    Task heads:\n    " + "    ".join(task_str.splitlines(True))

        child_str = "    " + pre_nn_str + pre_nn_edges_str + gnn_str + task_str
        child_str = "    ".join(child_str.splitlines(True))

        full_str = self.name + "\n" + "-" * (len(self.name) + 2) + "\n" + child_str

        return full_str

    @property
    def in_dim(self) -> int:
        r"""
        Returns the input dimension of the network
        """
        if self.pre_nn is not None:
            return self.pre_nn.in_dim
        else:
            return self.gnn.in_dim

    @property
    def out_dim(self) -> int:
        r"""
        Returns the output dimension of the network
        """
        return self.gnn.out_dim

    @property
    def out_dim_edges(self) -> int:
        r"""
        Returns the output dimension of the edges
        of the network.
        """
        if self.gnn.full_dims_edges is not None:
            return self.gnn.full_dims_edges[-1]
        return self.gnn.in_dim_edges

    @property
    def in_dim_edges(self) -> int:
        r"""
        Returns the input edge dimension of the network
        """
        return self.gnn.in_dim_edges


class GraphOutputNN(nn.Module, MupMixin):
    def __init__(
        self,
        in_dim: int,
        in_dim_edges: int,
        task_level: str,
        graph_output_nn_kwargs: Dict[str, Any],
    ):
        r"""
        Parameters:
            in_dim:
                Input feature dimensions of the layer

            in_dim_edges:
                Input edge feature dimensions of the layer
            task_level:
                graph/node/edge/nodepair depending on wether it is graph/node/edge/nodepair level task
            graph_output_nn_kwargs:
                key-word arguments to use for the initialization of the post-processing
                MLP network after the GNN, using the class `FeedForwardNN`.
        """
        super().__init__()
        self.task_level = task_level
        self._concat_last_layers = None
        self.in_dim = in_dim
        self.in_dim_edges = in_dim_edges
        self.graph_output_nn_kwargs = graph_output_nn_kwargs
        self.max_num_nodes_per_graph = None
        self.max_num_edges_per_graph = None
        self.map_task_level = {
            "node": "feat",
            "nodepair": "nodepair_feat",
            "graph": "graph_feat",
            "edge": "edge_feat",
        }

        if self.task_level == "nodepair":
            level_in_dim = 2 * self.in_dim
        elif self.task_level == "edge":
            level_in_dim = self.in_dim_edges
        elif self.task_level == "graph":
            self.global_pool_layer, self.out_pool_dim = self._parse_pooling_layer(
                self.in_dim, graph_output_nn_kwargs[self.task_level]["pooling"]
            )
            level_in_dim = self.out_pool_dim
        else:
            level_in_dim = self.in_dim

        if level_in_dim == 0:
            raise ValueError(f"Task head has an input dimension of 0.")
        # Initialize the post-processing neural net (applied after the gnn)
        name = graph_output_nn_kwargs[self.task_level].pop("name", "post-NN")
        filtered_graph_output_nn_kwargs = {
            k: v for k, v in graph_output_nn_kwargs[self.task_level].items() if k not in ["pooling", "in_dim"]
        }
        self.graph_output_nn = FeedForwardNN(
            in_dim=level_in_dim, name=name, **filtered_graph_output_nn_kwargs
        )

    def forward(self, g: Batch):
        """
        Parameters:
            g: pyg Batch graph
        Returns:
            h: Output features after applying graph_output_nn
        """
        # Check if at least one nodepair task is present
        if self.task_level == "nodepair":
            g["nodepair_feat"] = self.compute_nodepairs(
                node_feats=g["feat"],
                batch=g.batch,
                max_num_nodes=self.max_num_nodes_per_graph,
                drop_nodes_last_graph=is_running_on_ipu(),
            )
        # Check if at least one graph-level task is present
        if self.task_level == "graph":
            # pool features if the level is graph
            g["graph_feat"] = self._pool_layer_forward(g, g["feat"])

        h = g[self.map_task_level[self.task_level]]
        # Run the output network
        if self.concat_last_layers is None:
            h = self.graph_output_nn.forward(h)
        else:
            # Concatenate the output of the last layers according to `self._concat_last_layers``.
            # Useful for generating fingerprints
            h = [h]
            for ii in range(len(self.graph_output_nn.layers)):
                h.insert(0, self.graph_output_nn.layers[ii].forward(h[0]))  # Append in reverse
        return h

    def _parse_pooling_layer(
        self, in_dim: int, pooling: Union[str, List[str]], **kwargs
    ) -> Tuple[nn.Module, int]:
        r"""
        Return the pooling layer
        **This function is virtual, so it needs to be implemented by the child class.**

        Parameters:

            in_dim:
                The dimension at the input layer of the pooling

            pooling:
                The list of pooling layers to use. The accepted strings are:

                - "sum": `SumPooling`
                - "mean": `MeanPooling`
                - "max": `MaxPooling`
                - "min": `MinPooling`
                - "std": `StdPooling`
                - "s2s": `Set2Set`
                - "dir{int}": `DirPooling`

            kwargs:
                Kew-word arguments for the pooling layer initialization

        Return:
            pool_layer: Pooling layer module
            out_pool_dim: Output dimension of the pooling layer

        """
        return parse_pooling_layer_pyg(in_dim, pooling, **kwargs)

    def _pool_layer_forward(
        self,
        g: Batch,
        feat: torch.Tensor,
    ):
        r"""
        Apply the graph pooling layer, followed by the linear output layer.

        Parameters:

            g: pyg Batch graph on which the convolution is done

            feat (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the output size of the last Graph layer

        Returns:
            torch.Tensor[..., M, Din] or torch.Tensor[..., N, Din]:
                Node feature tensor, after convolution.
                `N` is the number of nodes, `M` is the number of graphs, `Dout` is the output dimension ``self.out_dim``
                If the pooling is `None`, the dimension is `N`, otherwise it is `M`

        """

        if len(self.global_pool_layer) > 0:
            pooled_feat = self.global_pool_layer(g, feat)
        else:
            pooled_feat = feat

        return pooled_feat

    def compute_nodepairs(
        self,
        node_feats: torch.Tensor,
        batch: torch.Tensor,
        max_num_nodes: int = None,
        fill_value: float = float("nan"),
        batch_size: int = None,
        drop_nodes_last_graph: bool = False,
    ) -> torch.Tensor:
        r"""
        Vectorized implementation of nodepair-level task:
        Parameters:
            node_feats: Node features
            batch: Batch vector
            max_num_nodes: The maximum number of nodes per graph
            fill_value: The value for invalid entries in the
                resulting dense output tensor. (default: :obj:`NaN`)
            batch_size: The batch size. (default: :obj:`None`)
            drop_nodes_last_graph: Whether to drop the nodes of the last graphs that exceed
                the `max_num_nodes_per_graph`. Useful when the last graph is a padding.
        Returns:
            result: concatenated node features of shape B * max_num_nodes * 2*h,
            where B is number of graphs, max_num_nodes is the chosen maximum number nodes, and h is the feature dim
        """
        dense_feat, mask, _ = to_dense_batch(
            node_feats,
            batch=batch,
            fill_value=fill_value,
            batch_size=batch_size,
            max_num_nodes_per_graph=max_num_nodes,
            drop_nodes_last_graph=drop_nodes_last_graph,
        )
        n = dense_feat.size(1)
        h_X = dense_feat[:, :, None].repeat(1, 1, n, 1)
        h_Y = dense_feat[:, None, :, :].repeat(1, n, 1, 1)

        nodepair_h = torch.cat((h_X + h_Y, torch.abs(h_X - h_Y)), dim=-1)
        upper_tri_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
        # Mask nodepair_h using upper_tri_mask
        batched_result = nodepair_h[:, upper_tri_mask, :]
        return batched_result

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension

        Returns:
            Dictionary with the kwargs to create the base model.
        """
        # For the post-nn network, all the dimension are divided
        graph_output_nn_kwargs = self.graph_output_nn.make_mup_base_kwargs(
            divide_factor=divide_factor, factor_in_dim=factor_in_dim
        )
        kwargs = {
            "pooling": self.graph_output_nn_kwargs[self.task_level]["pooling"],
            **graph_output_nn_kwargs,
        }
        return kwargs

    def drop_graph_output_nn_layers(self, num_layers_to_drop: int) -> None:
        r"""
        Remove the last layers of the model. Useful for Transfer Learning.
        Parameters:
            num_layers_to_drop: The number of layers to drop from the `self.graph_output_nn` network.
        """

        assert num_layers_to_drop >= 0
        assert num_layers_to_drop <= len(self.graph_output_nn.layers)

        if num_layers_to_drop > 0:
            self.graph_output_nn.layers = self.graph_output_nn.layers[:-num_layers_to_drop]

    def extend_graph_output_nn_layers(self, layers: nn.ModuleList):
        r"""
        Add layers at the end of the model. Useful for Transfer Learning.
        Parameters:
            layers: A ModuleList of all the layers to extend
        """

        assert isinstance(layers, nn.ModuleList)
        if len(self.graph_output_nn.layers) > 0:
            assert layers[0].in_dim == self.graph_output_nn.layers.out_dim[-1]

        self.graph_output_nn.extend(layers)

    def set_max_num_nodes_edges_per_graph(self, max_nodes: Optional[int], max_edges: Optional[int]) -> None:
        """
        Set the maximum number of nodes and edges for all gnn layers and encoder layers

        Parameters:
            max_nodes: Maximum number of nodes in the dataset.
                This will be useful for certain architecture, but ignored by others.

            max_edges: Maximum number of edges in the dataset.
                This will be useful for certain architecture, but ignored by others.
        """
        self.max_num_nodes_per_graph = max_nodes
        self.max_num_edges_per_graph = max_edges

    @property
    def concat_last_layers(self) -> Optional[Iterable[int]]:
        """
        Property to control the output of the `self.forward`.
        If set to a list of integer, the `forward` function will
        concatenate the output of different layers.

        If set to `None`, the output of the last layer is returned.

        NOTE: The indexes are inverted. 0 is the last layer, 1 is the second last, etc.
        """
        return self._concat_last_layers

    @concat_last_layers.setter
    def concat_last_layers(self, value: Union[Type[None], int, Iterable[int]]) -> None:
        """
        Set the property to control the output of the `self.forward`.
        If set to a list of integer, the `forward` function will
        concatenate the output of different layers.
        If a single integer is provided, it will output that specific layer.

        If set to `None`, the output of the last layer is returned.

        NOTE: The indexes are inverted. 0 is the last layer, 1 is the second last, etc.

        Parameters:
            value: Output layers to concatenate, in reverse order (`0` is the last layer)
        """
        if (value is not None) and not isinstance(value, Iterable):
            value = [value]
        self._concat_last_layers = value

    @property
    def out_dim(self) -> int:
        r"""
        Returns the output dimension of the network
        """
        return self.graph_output_nn.out_dim


class TaskHeads(nn.Module, MupMixin):
    def __init__(
        self,
        in_dim: int,
        in_dim_edges: int,
        task_heads_kwargs: Dict[str, Any],
        graph_output_nn_kwargs: Dict[str, Any],
        last_layer_is_readout: bool = True,
    ):
        r"""
        Class that groups all multi-task output heads together to provide the task-specific outputs.
        Parameters:
            in_dim:
                Input feature dimensions of the layer

            in_dim_edges:
                Input edge feature dimensions of the layer
            last_layer_is_readout: Whether the last layer should be treated as a readout layer.
                Allows to use the `mup.MuReadout` from the muTransfer method
            task_heads_kwargs:
                This argument is a list of dictionaries corresponding to the arguments for a FeedForwardNN.
                Each dict of arguments is used to initialize a task-specific MLP.
            graph_output_nn_kwargs:
                key-word arguments to use for the initialization of the post-processing
                MLP network after the GNN, using the class `FeedForwardNN`.
        """
        super().__init__()
        self.last_layer_is_readout = last_layer_is_readout
        self.task_heads_kwargs = deepcopy(task_heads_kwargs)
        self.graph_output_nn_kwargs = deepcopy(graph_output_nn_kwargs)
        self.task_levels = {head_kwargs["task_level"] for _, head_kwargs in self.task_heads_kwargs.items()}
        self.in_dim = in_dim
        self.in_dim_edges = in_dim_edges
        self.task_heads = nn.ModuleDict()
        self.graph_output_nn = nn.ModuleDict()
        self._check_bad_arguments()

        for task_name, head_kwargs in self.task_heads_kwargs.items():
            task_level = self.task_heads_kwargs[task_name].get("task_level")
            self.graph_output_nn[task_level] = GraphOutputNN(
                in_dim=self.in_dim,
                in_dim_edges=self.in_dim_edges,
                task_level=task_level,
                graph_output_nn_kwargs=self.graph_output_nn_kwargs,
            )
            head_kwargs.setdefault("name", f"NN-{task_name}")
            head_kwargs.setdefault("last_layer_is_readout", last_layer_is_readout)
            # Create a new dictionary without the task_level key-value pair,
            # and pass it while initializing the FeedForwardNN instance for tasks
            filtered_kwargs = {k: v for k, v in head_kwargs.items() if k != "task_level"}
            filtered_kwargs["in_dim"] = self.graph_output_nn_kwargs[task_level]["out_dim"]
            self.task_heads[task_name] = FeedForwardNN(**filtered_kwargs)

    def forward(self, g: Batch) -> Dict[str, torch.Tensor]:
        r"""
        forward function of the task head
        Parameters:
            g: pyg Batch graph
        Returns:
            task_head_outputs: Return a dictionary: Dict[task_name, Tensor]
        """
        features = {task_level: self.graph_output_nn[task_level](g) for task_level in self.task_levels}

        task_head_outputs = {}
        for task_name, head in self.task_heads.items():
            task_level = self.task_heads_kwargs[task_name].get(
                "task_level", None
            )  # Get task_level without modifying head_kwargs
            task_head_outputs[task_name] = head.forward(features[task_level])

        return task_head_outputs

    def make_mup_base_kwargs(self, divide_factor: float = 2.0, factor_in_dim: bool = False) -> Dict[str, Any]:
        """
        Create a 'base' model to be used by the `mup` or `muTransfer` scaling of the model.
        The base model is usually identical to the regular model, but with the
        layers width divided by a given factor (2 by default)

        Parameter:
            divide_factor: Factor by which to divide the width.
            factor_in_dim: Whether to factor the input dimension

        Returns:
            kwargs: Dictionary of arguments to be used to initialize the base model
        """
        graph_output_nn_kwargs = {}
        for task_level, graph_output_nn in self.graph_output_nn.items():
            graph_output_nn: GraphOutputNN
            graph_output_nn_kwargs[task_level] = graph_output_nn.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=factor_in_dim
            )

        task_heads_kwargs = {}
        for task_name, task_nn in self.task_heads.items():
            task_nn: FeedForwardNN
            task_heads_kwargs[task_name] = task_nn.make_mup_base_kwargs(
                divide_factor=divide_factor, factor_in_dim=factor_in_dim
            )
            task_heads_kwargs[task_name]["task_level"] = self.task_heads_kwargs[task_name]["task_level"]
        kwargs = dict(
            in_dim=self.in_dim,
            last_layer_is_readout=self.last_layer_is_readout,
            task_heads_kwargs=task_heads_kwargs,
            graph_output_nn_kwargs=graph_output_nn_kwargs,
        )
        return kwargs

    def set_max_num_nodes_edges_per_graph(self, max_nodes: Optional[int], max_edges: Optional[int]) -> None:
        """
        Set the maximum number of nodes and edges for all gnn layers and encoder layers

        Parameters:
            max_nodes: Maximum number of nodes in the dataset.
                This will be useful for certain architecture, but ignored by others.

            max_edges: Maximum number of edges in the dataset.
                This will be useful for certain architecture, but ignored by others.
        """
        for graph_output_nn in self.graph_output_nn.values():
            graph_output_nn: GraphOutputNN
            graph_output_nn.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)

        for task_head in self.task_heads.values():
            task_head: FeedForwardNN
            for layer in task_head.layers:
                if isinstance(layer, BaseGraphStructure):
                    layer.max_num_nodes_per_graph = max_nodes
                    layer.max_num_edges_per_graph = max_edges

    @property
    def out_dim(self) -> Dict[str, int]:
        r"""
        Returns the output dimension of each task head
        """
        return {task_name: head.out_dim for task_name, head in self.task_heads.items()}

    def __repr__(self):
        r"""
        Returns a string representation of the task heads
        """
        task_repr = []
        for head, net in self.task_heads.items():
            task_repr.append(head + ": " + net.__repr__())
        return "\n".join(task_repr)

    def _check_bad_arguments(self):
        r"""
        Raise comprehensive errors if the arguments seem wrong
        """
        for task_name, head_kwargs in self.task_heads_kwargs.items():
            task_level = self.task_heads_kwargs[task_name].get("task_level", None)
            if task_level is None:
                raise ValueError("task_level must be specified for each task head.")
            if task_level not in ["node", "edge", "graph", "nodepair"]:
                raise ValueError(f"task_level {task_level} is not supported.")
