from torch import Tensor, nn
import torch
from typing import Iterable, List, Dict, Tuple, Union, Callable, Any, Optional, Type
import inspect
from copy import deepcopy

from dgl import DGLGraph
from torch_geometric.data import Data

from goli.nn.base_layers import FCLayer, get_activation, get_norm
from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.residual_connections import (
    ResidualConnectionBase,
    ResidualConnectionWeighted,
    ResidualConnectionRandom,
)

from goli.nn.encoders import laplace_pos_encoder, mlp_encoder, signnet_pos_encoder

PE_ENCODERS_DICT = {
    "laplacian_pe": laplace_pos_encoder.LapPENodeEncoder,
    "mlp": mlp_encoder.MLPEncoder,
    "signnet": signnet_pos_encoder.SignNetNodeEncoder,
}


class FeedForwardNN(nn.Module):
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
                hidden layers to use. If `hidden_dims` is a `list`, `depth` must
                be `None`.

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

        """

        super().__init__()

        # Set the class attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        if isinstance(hidden_dims, int):
            self.hidden_dims = [hidden_dims] * (depth - 1)
        else:
            self.hidden_dims = list(hidden_dims)
            assert depth is None
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

        # Parse the layer and residuals
        from goli.utils.spaces import LAYERS_DICT, RESIDUALS_DICT

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
            if ii == self.depth - 1:
                this_activation = self.last_activation
                this_norm = self.last_normalization
                this_dropout = self.last_dropout

            # Create the layer
            self.layers.append(
                self.layer_class(
                    in_dim=this_in_dim,
                    out_dim=this_out_dim,
                    activation=this_activation,
                    dropout=this_dropout,
                    normalization=this_norm,
                    **self.layer_kwargs,
                )
            )

            if ii < len(residual_out_dims):
                this_in_dim = residual_out_dims[ii]

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
        h_prev = None

        # Apply a normalization before the first layer
        if self.first_normalization is not None:
            h = self.first_normalization(h)

        # Apply all neural network layers
        for ii, layer in enumerate(self.layers):
            h = layer.forward(h)
            if ii < len(self.layers) - 1:
                h, h_prev = self.residual_layer.forward(h, h_prev, step_idx=ii)

        return h

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        class_str = f"{self.name}(depth={self.depth}, {self.residual_layer})\n    "
        layer_str = f"[{self.layer_class.__name__}[{' -> '.join(map(str, self.full_dims))}]"

        return class_str + layer_str


class FeedForwardGraphBase(FeedForwardNN):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
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
        pooling: Union[List[str], List[Callable]] = ["sum"],
        name: str = "GNN",
        layer_kwargs: Optional[Dict] = None,
        virtual_node: str = "none",
    ):
        r"""
        **Astract class, must be inherited to override the following methods:**
        - `_graph_layer_forward`
        - `_parse_virtual_node_class`
        - `_parse_pooling_layer`
        - `_get_node_feats
        - `_get_edge_feats`
        - `_set_node_feats`
        - `_set_edge_feats`

        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different graph neural networks
        layers. Any layer must inherit from `goli.nn.base_graph_layer.BaseGraphStructure`
        or `goli.nn.base_graph_layer.BaseGraphLayer`.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            hidden_dims:
                List of dimensions in the hidden layers.
                Be careful, the "simple" residual type only supports
                hidden dimensions of the same value.

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

            pooling:
                The pooling types to use. Multiple pooling can be used, and their
                results will be concatenated.
                For node feature predictions, use `["none"]`.
                For graph feature predictions see `goli.nn.dgl_layers.pooling.parse_pooling_layer`.
                The list must either contain Callables, or the string below

                - "none": No pooling is applied
                - "sum": `SumPooling`
                - "mean": `MeanPooling`
                - "max": `MaxPooling`
                - "min": `MinPooling`
                - "std": `StdPooling`
                - "s2s": `Set2Set`

            name:
                Name attributed to the current network, for display and printing
                purposes.

            layer_type:
                The type of layers to use in the network.
                A class that inherits from `goli.nn.dgl_layers.BaseDGLLayer`,
                or one of the following strings

                - "dgl:gcn": GCNDgl
                - "dgl:gin": GINDgl
                - "dgl:gat": GATDgl
                - "dgl:gated-gcn": GatedGCNDgl
                - "dgl:pna-conv": PNAConvolutionalDgl
                - "dgl:pna-msgpass": PNAMessagePassingDgl
                - "dgl:dgn-conv": DGNConvolutionalDgl
                - "dgl:dgn-msgpass": DGNMessagePassingDgl
                - "pyg:gin": GINConvPyg
                - "pyg:gine": GINEConvPyg
                - "pyg:gated-gcn": GatedGCNPyg
                - "pyg:pna-msgpass": PNAMessagePassingPyg

            layer_kwargs:
                The arguments to be used in the initialization of the layer provided by `layer_type`

            virtual_node:
                A string associated to the type of virtual node to use,
                either `None`, "none", "mean", "sum", "max", "logsum".
                See `goli.nn.dgl_layers.VirtualNode`.

                The virtual node will not use any residual connection if `residual_type`
                is "none". Otherwise, it will use a simple ResNet like residual
                connection.

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
        self.pooling = pooling

        self.virtual_node_class = self._parse_virtual_node_class()
        self.first_normalization_edges = get_norm(first_normalization, dim=in_dim_edges)

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
        )

    def _check_bad_arguments(self):
        r"""
        Raise comprehensive errors if the arguments seem wrong
        """
        super()._check_bad_arguments()
        if (
            (self.in_dim_edges > 0) or (self.full_dims_edges is not None)
        ) and not self.layer_class.layer_supports_edges:
            raise ValueError(f"Cannot use edge features with class `{self.layer_class}`")

    def _parse_virtual_node_class(self) -> type:
        r"""
        Virtual method to parse the VirtualNode class. Must be inherited.

        Should simply return the class of the virtual node that works
        with the specified graph. Example below.
        `return goli.nn.dgl_layers.pooling_dgl.VirtualNodeDgl`.
        """
        raise NotImplementedError

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

            if ii == self.depth - 1:
                this_activation = self.last_activation

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
                    **self.layer_kwargs,
                    **this_edge_kwargs,
                )
            )

            # Create the Virtual Node layer, except at the last layer
            if ii < len(residual_out_dims):
                self.virtual_node_layers.append(
                    self.virtual_node_class(
                        dim=this_out_dim * self.layers[-1].out_dim_factor,
                        activation=this_activation,
                        dropout=this_dropout,
                        normalization=this_norm,
                        bias=True,
                        vn_type=self.virtual_node,
                        residual=self.residual_type is not None,
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
        self.global_pool_layer, out_pool_dim = self._parse_pooling_layer(layer_out_dims[-1], self.pooling)

        # Output linear layer
        self.out_linear = FCLayer(
            in_dim=out_pool_dim,
            out_dim=self.out_dim,
            activation="none",
            dropout=self.dropout,
            normalization=self.normalization,
        )

    def _pool_layer_forward(self, g, h):
        r"""
        Apply the graph pooling layer, followed by the linear output layer.

        Parameters:

            g: dgl.DGLGraph
                graph on which the convolution is done

            h (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the output size of the last DGL layer

        Returns:

            torch.Tensor[..., M, Din] or torch.Tensor[..., N, Din]:
                Node feature tensor, after convolution.
                `N` is the number of nodes, `M` is the number of graphs, `Dout` is the output dimension ``self.out_dim``
                If the pooling is `None`, the dimension is `N`, otherwise it is `M`

        """

        if len(self.global_pool_layer) > 0:
            pooled_h = self.global_pool_layer(g, h)
        else:
            pooled_h = h

        pooled_h = self.out_linear(pooled_h)

        return pooled_h

    def _graph_layer_forward(
        self,
        layer: BaseGraphModule,
        g,
        h: torch.Tensor,
        e: Union[torch.Tensor, None],
        h_prev: Union[torch.Tensor, None],
        e_prev: Union[torch.Tensor, None],
        step_idx: int,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Apply the *i-th* graph layer, where *i* is the index given by `step_idx`.
        The layer is applied differently depending if there are edge features or not.

        Then, the residual is also applied on both the features and the edges (if applicable)

        **This function is virtual, so it needs to be implemented by the child class.**

        Parameters:

            layer:
                The layer used for the convolution

            g:
                batched graphs on which the convolution is done

            h (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            e (torch.Tensor[..., N, Ein]):
                Edge feature tensor, before convolution.
                `N` is the number of nodes, `Ein` is the input edge features

            h_prev:
                Node feature of the previous residual connection, or `None`

            e_prev:
                Edge feature of the previous residual connection, or `None`

            step_idx:
                The current step idx in the forward loop

        Returns:

            h (torch.Tensor[..., N, Dout]):
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            e:
                Edge feature tensor, after convolution and residual.
                `N` is the number of nodes, `Ein` is the input edge features

            h_prev:
                Node feature tensor to be used at the next residual connection, or `None`

            e_prev:
                Edge feature tensor to be used at the next residual connection, or `None`

        """

        raise NotImplementedError("Virtual method must be overwritten by child class")

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
        raise NotImplementedError("Virtual method must be overwritten by child class")

    def _virtual_node_forward(
        self, g: Union[DGLGraph, Data], h: torch.Tensor, vn_h: torch.Tensor, step_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the *i-th* virtual node layer, where *i* is the index given by `step_idx`.

        Parameters:

            g:
                graph on which the convolution is done

            h (torch.Tensor[..., N, Din]):
                Node feature tensor, before convolution.
                `N` is the number of nodes, `Din` is the input features

            vn_h (torch.Tensor[..., M, Din]):
                Graph feature of the previous virtual node, or `None`
                `M` is the number of graphs, `Din` is the input features
                It is added to the result after the MLP, as a residual connection

            step_idx:
                The current step idx in the forward loop

        Returns:

            `h = torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution and residual.
                `N` is the number of nodes, `Dout` is the output features of the layer and residual

            `vn_h = torch.Tensor[..., M, Dout]`:
                Graph feature tensor to be used at the next virtual node, or `None`
                `M` is the number of graphs, `Dout` is the output features

        """

        if step_idx == 0:
            vn_h = 0.0
        if step_idx < len(self.virtual_node_layers):
            h, vn_h = self.virtual_node_layers[step_idx].forward(g=g, h=h, vn_h=vn_h)

        return h, vn_h

    def forward(self, g) -> torch.Tensor:
        r"""
        Apply the full graph neural network on the input graph and node features.

        Parameters:

            g:
                batched graphs on which the convolution is done with the keys:

                - `"h"`: torch.Tensor[..., N, Din]
                  Node feature tensor, before convolution.
                  `N` is the number of nodes, `Din` is the input features

                - `"edge_attr"` (torch.Tensor[..., N, Ein]):
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
        h_prev = None
        e_prev = None
        vn_h = 0
        h = self._get_node_feats(g, key="h")
        e = self._get_edge_feats(g, key="edge_attr")

        # Apply the normalization before the first network layers
        if self.first_normalization is not None:
            h = self.first_normalization(h)
        if self.first_normalization_edges is not None:
            e = self.first_normalization_edges(e)

        # Apply the forward loop of the layers, residuals and virtual nodes
        for ii, layer in enumerate(self.layers):
            h, e, h_prev, e_prev = self._graph_layer_forward(
                layer=layer, g=g, h=h, e=e, h_prev=h_prev, e_prev=e_prev, step_idx=ii
            )
            h, vn_h = self._virtual_node_forward(g=g, h=h, vn_h=vn_h, step_idx=ii)

        pooled_h = self._pool_layer_forward(g=g, h=h)

        return pooled_h

    def _get_node_feats(self, g, key: str = "h") -> Tensor:
        """
        Get the node features of a graph `g`.
        ***Virtual method, must be implemented in child class.***

        Parameters:
            g: graph
            key: key associated to the node features
        """
        raise NotImplementedError("Virtual method must be overwritten by child class")

    def _get_edge_feats(self, g, key: str = "edge_attr") -> Tensor:
        """
        Get the edge features of a graph `g`.
        ***Virtual method, must be implemented in child class.***

        Parameters:
            g: graph
            key: key associated to the edge features
        """
        raise NotImplementedError("Virtual method must be overwritten by child class")

    def _set_node_feats(self, g: Any, node_feats: Tensor, key: str = "h") -> Any:
        """
        Set the node features of a graph `g`, and return the graph.
        ***Virtual method, must be implemented in child class.***

        Parameters:
            g: graph
            key: key associated to the node features
        """
        raise NotImplementedError("Virtual method must be overwritten by child class")
        return g

    def _set_edge_feats(self, g: Any, edge_feats: Tensor, key: str = "edge_attr") -> Any:
        """
        Set the edge features of a graph `g`, and return the graph.
        ***Virtual method, must be implemented in child class.***

        Parameters:
            g: graph
            key: key associated to the edge features
        """
        raise NotImplementedError("Virtual method must be overwritten by child class")
        return g

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        class_str = f"{self.name}(depth={self.depth}, {self.residual_layer})\n    "
        layer_str = f"{self.layer_class.__name__}[{' -> '.join(map(str, self.full_dims))}]\n    "
        pool_str = f"-> Pooling({self.pooling})"
        out_str = f" -> {self.out_linear}"

        return class_str + layer_str + pool_str + out_str


class FullGraphNetwork(nn.Module):
    def __init__(
        self,
        gnn_kwargs: Dict[str, Any],
        pre_nn_kwargs: Optional[Dict[str, Any]] = None,
        pre_nn_edges_kwargs: Optional[Dict[str, Any]] = None,
        pe_encoders_kwargs: Optional[Dict[str, Any]] = None,
        post_nn_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_to_average: int = 1,
        name: str = "DGL_GNN",
    ):
        r"""
        Class that allows to implement a full graph neural network architecture,
        including the pre-processing MLP and the post processing MLP.

        Parameters:

            gnn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                GNN network using the class `FeedForwardDGL`.
                It must respect the following criteria:

                - gnn_kwargs["in_dim"] must be equal to pre_nn_kwargs["out_dim"]
                - gnn_kwargs["out_dim"] must be equal to post_nn_kwargs["in_dim"]

            pe_encoders_kwargs:
                key-word arguments to use for the initialization of all positional encoding encoders
                can use the class PE_ENCODERS_DICT: "la_encoder"(tested) , "mlp_encoder" (not tested), "signnet_encoder" (not tested)

            pre_nn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the node features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            pre_nn_edges_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the edge features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            post_nn_kwargs:
                key-word arguments to use for the initialization of the post-processing
                MLP network after the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a post-processing MLP.

            num_inference_to_average:
                Number of inferences to average at val/test time. This is used to avoid the noise introduced
                by positional encodings with sign-flips. In case no such encoding is given,
                this parameter is ignored.
                NOTE: The inference time will be slowed-down proportionaly to this parameter.

            name:
                Name attributed to the current network, for display and printing
                purposes.
        """

        super().__init__()
        self.name = name
        self.num_inference_to_average = num_inference_to_average
        self._concat_last_layers = None
        self.pre_nn, self.post_nn, self.pre_nn_edges = None, None, None

        self.pe_encoders = self._initialize_positional_encoders(pe_encoders_kwargs)

        # Initialize the pre-processing neural net for nodes (applied directly on node features)
        if pre_nn_kwargs is not None:
            name = pre_nn_kwargs.pop("name", "pre-NN")
            self.pre_nn = FeedForwardNN(**pre_nn_kwargs, name=name)
            next_in_dim = self.pre_nn.out_dim
            gnn_kwargs.setdefault("in_dim", next_in_dim)
            assert next_in_dim == gnn_kwargs["in_dim"], "Inconsistent input/output dimensions"

        # Initialize the pre-processing neural net for edges (applied directly on edge features)
        if pre_nn_edges_kwargs is not None:
            name = pre_nn_edges_kwargs.pop("name", "pre-NN-edges")
            self.pre_nn_edges = FeedForwardNN(**pre_nn_edges_kwargs, name=name)
            next_in_dim = self.pre_nn_edges.out_dim
            gnn_kwargs.setdefault("in_dim_edges", next_in_dim)
            assert next_in_dim == gnn_kwargs["in_dim_edges"], "Inconsistent input/output dimensions"

        # Initialize the graph neural net (applied after the pre_nn)
        name = gnn_kwargs.pop("name", "GNN")
        gnn_class = self._parse_feed_forward_gnn(gnn_kwargs)
        self.gnn = gnn_class(**gnn_kwargs, name=name)
        next_in_dim = self.gnn.out_dim

        # Initialize the post-processing neural net (applied after the gnn)
        if post_nn_kwargs is not None:
            name = post_nn_kwargs.pop("name", "post-NN")
            post_nn_kwargs.setdefault("in_dim", next_in_dim)
            self.post_nn = FeedForwardNN(**post_nn_kwargs, name=name)
            assert next_in_dim == self.post_nn.in_dim, "Inconsistent input/output dimensions"

    def _initialize_positional_encoders(self, pe_encoders_kwargs: Dict[str, Any]) -> Optional[nn.ModuleDict]:
        """
        Initialize the positional encoders for each positional/structural encodings.

        TODO: Currently only supports PE/SE on the nodes. Need to add edges.
        """
        pe_encoders = None

        if pe_encoders_kwargs is not None:
            pe_encoders = nn.ModuleDict()

            # Pooling options here for pe encoders
            self.pe_pool = pe_encoders_kwargs["pool"]
            pe_out_dim = pe_encoders_kwargs["out_dim"]
            in_dim_dict = pe_encoders_kwargs["in_dims"]

            # Loop every positional encoding to assign it
            for encoder_name, encoder_kwargs in pe_encoders_kwargs["encoders"].items():
                encoder_kwargs = deepcopy(encoder_kwargs)
                encoder_type = encoder_kwargs.pop("encoder_type")
                encoder = PE_ENCODERS_DICT[encoder_type]

                # Get the keys associated to in_dim
                this_in_dims = {}
                for key, dim in in_dim_dict.items():
                    if isinstance(key, str) and key.startswith(f"{encoder_name}/"):
                        key_name = "in_dim_" + key[len(encoder_name) + 1 :]
                        this_in_dims[key_name] = dim
                assert (
                    len(this_in_dims) > 0
                ), f"Non-matching in_dim. Provided: '{encoder_name}/'. Available keys: {in_dim_dict.keys()}"

                # Parse the in_dims based on Encoder's signature
                accepted_keys = inspect.signature(encoder).parameters.keys()
                if all([key in accepted_keys for key in this_in_dims.keys()]):
                    pass
                elif "in_dim" in accepted_keys:
                    if len(set(this_in_dims.values())) == 1:
                        this_in_dims = {"in_dim": list(this_in_dims.values())[0]}
                    else:
                        raise ValueError(
                            f"All `in_dims` must be equal for encoder {encoder_name}. Provided: {this_in_dims}"
                        )
                else:
                    raise ValueError(
                        f"`in_dim` not understood for encoder {encoder_name}. Provided: {this_in_dims}. Accepted keys are: {accepted_keys}"
                    )

                # Initialize the pe_encoder layer
                pe_encoders[encoder_name] = encoder(out_dim=pe_out_dim, **this_in_dims, **encoder_kwargs)

        return pe_encoders

    @staticmethod
    def _parse_feed_forward_gnn(gnn_kwargs):
        """
        Returns either `FeedForwardDGL` or `FeedForwardPyg`.
        """

        layer_type = gnn_kwargs.get("layer_type")

        # Get the layer name
        layer_name = layer_type
        if not isinstance(layer_name, str):
            if inspect.isclass(layer_name):
                layer_name = layer_name.__name__
            else:
                raise TypeError("`layer_type` should be `str` or class")
        layer_name = layer_name.lower()

        # Return the right FeedForward class
        if layer_name.startswith("dgl:") or layer_name.endswith("dgl"):
            # Importing here to avoid circular imports
            from goli.nn.architectures import FeedForwardDGL

            return FeedForwardDGL
        if layer_name.startswith("pyg:") or layer_name.endswith("pyg"):
            # Importing here to avoid circular imports
            from goli.nn.architectures import FeedForwardPyg

            return FeedForwardPyg
        else:
            raise TypeError(f"Can't recognize if `{layer_name}` uses Pyg or DGL")

    def _check_bad_arguments(self):
        r"""
        Raise comprehensive errors if the arguments seem wrong
        """
        if self.pre_nn is not None:
            if self.pre_nn["out_dim"] != self.gnn["in_dim"]:
                raise ValueError(
                    f"`self.pre_nn.out_dim` must be equal to `self.gnn.in_dim`."
                    + 'Provided" {self.pre_nn.out_dim} and {self.gnn.in_dim}'
                )

        if self.post_nn is not None:
            if self.gnn["out_dim"] != self.post_nn["in_dim"]:
                raise ValueError(
                    f"`self.gnn.out_dim` must be equal to `self.post_nn.in_dim`."
                    + 'Provided" {self.gnn.out_dim} and {self.post_nn.in_dim}'
                )

    def drop_post_nn_layers(self, num_layers_to_drop: int) -> None:
        r"""
        Remove the last layers of the model. Useful for Transfer Learning.

        Parameters:
            num_layers_to_drop: The number of layers to drop from the `self.post_nn` network.

        """

        assert num_layers_to_drop >= 0
        assert num_layers_to_drop <= len(self.post_nn.layers)

        if num_layers_to_drop > 0:
            self.post_nn.layers = self.post_nn.layers[:-num_layers_to_drop]

    def extend_post_nn_layers(self, layers: nn.ModuleList):
        r"""
        Add layers at the end of the model. Useful for Transfer Learning.

        Parameters:
            layers: A ModuleList of all the layers to extend

        """

        assert isinstance(layers, nn.ModuleList)
        if len(self.post_nn.layers) > 0:
            assert layers[0].in_dim == self.post_nn.layers.out_dim[-1]

        self.post_nn.extend(layers)

    def forward(self, g: Any) -> Tensor:
        r"""
        Apply the pre-processing neural network, the graph neural network,
        and the post-processing neural network on the graph features.

        Parameters:

            g:
                graph on which the convolution is done.
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
                `Dout` is the output dimension ``self.post_nn.out_dim``
                If the `self.gnn.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        return self._forward(g)

    def _forward(self, g: Any) -> Tensor:
        h = self.gnn._get_node_feats(g, key="feat")
        e = self.gnn._get_edge_feats(g, key="edge_feat")

        # Node-wise positional encoding, concatenated to node features.
        pe_node = self.forward_node_positional_encoding(g)
        if pe_node is not None:
            h = torch.cat((h, pe_node), dim=-1)

        # TODO: Add edge-wise positional encoding
        # pe_node = self.forward_edge_positional_encoding(g)

        # Set the node and edge features before running the GNN
        g = self.gnn._set_node_feats(g, h.to(self.dtype), key="h")
        if e is not None:
            e = e.to(self.dtype)
        g = self.gnn._set_edge_feats(g, e, key="edge_attr")

        # Run the pre-processing network on node features
        if self.pre_nn is not None:
            h = self.gnn._get_node_feats(g, key="h")
            h = self.pre_nn.forward(h)
            g = self.gnn._set_node_feats(g, h, key="h")

        # Run the pre-processing network on edge features
        # If there are no edges, skip the forward and change the dimension of e
        if self.pre_nn_edges is not None:
            e = self.gnn._get_edge_feats(g, key="edge_attr")
            if torch.prod(torch.as_tensor(e.shape[:-1])) == 0:
                e = torch.zeros(
                    list(e.shape[:-1]) + [self.pre_nn_edges.out_dim], device=e.device, dtype=e.dtype
                )
            else:
                e = self.pre_nn_edges.forward(e)
            g = self.gnn._set_edge_feats(g, e, key="edge_attr")

        # Run the graph neural network
        h = self.gnn.forward(g)

        # Run the output network
        if self.post_nn is not None:
            if self.concat_last_layers is None:
                h = self.post_nn.forward(h)
            else:
                # Concatenate the output of the last layers according to `self._concat_last_layers``.
                # Useful for generating fingerprints
                h = [h]
                for ii in range(len(self.post_nn.layers)):
                    h.insert(0, self.post_nn.layers[ii].forward(h[0]))  # Append in reverse order
                h = torch.cat([h[ii] for ii in self._concat_last_layers], dim=-1)

        return h

    def forward_node_positional_encoding(self, g: Any) -> Optional[Tensor]:
        """
        Forward pass for the positional encodings (PE) on the nodes,
        with each PE having it's own encoder defined in `self.pe_encoders`.

        Parameters:
            g: graph containing the node positional encodings

        Returns:
            pe_node_pooled: The positional / structural encodings go through
            encoders, then are pooled together

        """

        # Return None if no positional encoders
        if (self.pe_encoders is None) or len(self.pe_encoders) == 0:
            return None

        encoder_outs = []
        # Run every node positional-encoder
        for name, encoder in self.pe_encoders.items():
            keys = encoder.on_keys
            encoder_inputs = {}
            for key in keys:
                encoder_inputs[key] = self.gnn._get_node_feats(g, key=f"{name}/{key}").to(self.dtype)
            encoder_outs.append(
                encoder(**encoder_inputs)["node"]
            )  # TODO: Avoid repeated call to encoder when using edges

        # Pool the node positional encodings
        pe_outs = torch.stack(encoder_outs, dim=-1)
        pe_node_pooled = self.forward_simple_pooling(pe_outs, pooling=self.pe_pool, dim=-1)

        return pe_node_pooled

    def forward_edge_positional_encoding(self, g):
        # TODO: Implement edge-wise positional encoding
        raise NotImplementedError("Not yet implemented")

    def forward_simple_pooling(self, h: Tensor, pooling: str, dim: int) -> Tensor:
        """
        Apply sum, mean, or max pooling on a Tensor.
        """

        if pooling == "sum":
            pooled = torch.sum(h, dim=dim)
        elif pooling == "mean":
            pooled = torch.mean(h, dim=dim)
        elif pooling == "max":
            pooled = torch.max(h, dim=dim)
        else:
            raise Exception(f"Pooling method `{self.pe_pool}` is not defined")
        return pooled

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

    def __repr__(self) -> str:
        r"""
        Controls how the class is printed
        """
        pre_nn_str, post_nn_str, pre_nn_edges_str = "", "", ""
        if self.pre_nn is not None:
            pre_nn_str = self.pre_nn.__repr__() + "\n\n"
        if self.pre_nn_edges is not None:
            pre_nn_edges_str = self.pre_nn_edges.__repr__() + "\n\n"
        gnn_str = self.gnn.__repr__() + "\n\n"
        if self.post_nn is not None:
            post_nn_str = self.post_nn.__repr__()

        child_str = "    " + pre_nn_str + pre_nn_edges_str + gnn_str + post_nn_str
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
        if self.post_nn is not None:
            return self.post_nn.out_dim
        else:
            return self.gnn.out_dim

    @property
    def in_dim_edges(self) -> int:
        r"""
        Returns the input edge dimension of the network
        """
        return self.gnn.in_dim_edges

    @property
    def dtype(self) -> torch.dtype:
        """
        Get the dtype of the current network, based on the weights of linear layers within the GNN
        """
        return self.gnn.out_linear.linear.weight.dtype


class FullGraphSiameseNetwork(FullGraphNetwork):
    def __init__(self, pre_nn_kwargs, gnn_kwargs, post_nn_kwargs, dist_method, name="Siamese_DGL_GNN"):

        # Initialize the parent nn.Module
        super().__init__(
            pre_nn_kwargs=pre_nn_kwargs,
            gnn_kwargs=gnn_kwargs,
            post_nn_kwargs=post_nn_kwargs,
            name=name,
        )

        self.dist_method = dist_method.lower()

    def forward(self, graphs):
        graph_1, graph_2 = graphs

        out_1 = super().forward(graph_1)
        out_2 = super().forward(graph_2)

        if self.dist_method == "manhattan":
            # Normalized L1 distance
            out_1 = out_1 / torch.mean(out_1.abs(), dim=-1, keepdim=True)
            out_2 = out_2 / torch.mean(out_2.abs(), dim=-1, keepdim=True)
            dist = torch.abs(out_1 - out_2)
            out = torch.mean(dist, dim=-1)

        elif self.dist_method == "euclidean":
            # Normalized Euclidean distance
            out_1 = out_1 / torch.norm(out_1, dim=-1, keepdim=True)
            out_2 = out_2 / torch.norm(out_2, dim=-1, keepdim=True)
            out = torch.norm(out_1 - out_2, dim=-1)
        elif self.dist_method == "cosine":
            # Cosine distance
            out = torch.sum(out_1 * out_2, dim=-1) / (torch.norm(out_1, dim=-1) * torch.norm(out_2, dim=-1))
        else:
            raise ValueError(f"Unsupported `dist_method`: {self.dist_method}")

        return out


class TaskHead(FeedForwardNN):
    def __init__(
        self,
        task_name: str,  # The name matters for per-task analysis
        in_dim: int,
        out_dim: int,
        hidden_dims: Union[List[int], int],  # Should this only be List? See FeedForwardNN vs. FeedForwardDGL
        depth: Optional[int] = None,
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        last_dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        last_normalization: Union[str, Callable] = "none",
        residual_type: str = "none",
        residual_skip_steps: int = 1,
        name: str = "LNN",
        layer_type: Union[str, nn.Module] = "fc",
        layer_kwargs: Optional[Dict] = None,
    ):
        r"""
        This class instantiates a task head, and it is identical to the FeedForwardNN class with the addition of a `task_name` attribute.
        Parameters:
            task_name:
                The name of the task for which the current output head performs predictions.
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
                hidden layers to use. If `hidden_dims` is a `list`, `depth` must
                be `None`.
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
        """

        self.task_name = task_name

        # Initialize the FeedForwardNN
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            depth=depth,
            activation=activation,
            last_activation=last_activation,
            normalization=normalization,
            last_normalization=last_normalization,
            residual_type=residual_type,
            residual_skip_steps=residual_skip_steps,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            last_dropout=last_dropout,
            layer_kwargs=layer_kwargs,
        )


class TaskHeads(nn.Module):
    def __init__(
        self,
        in_dim: int,
        task_heads_kwargs_list: List[Dict[str, Any]],
    ):
        r"""
        Class that groups all multi-task output heads together to provide the task-specific outputs.
        Parameters:
            in_dim:
                Input feature dimensions of the layer
            task_heads_kwargs_list:
                This argument is a list of dictionaries corresponding to the arguments for a TaskHead. Each dict of arguments is used to
                initialize a task-specific MLP.
        """

        super().__init__()

        self.in_dim = in_dim

        self.task_heads = nn.ModuleDict()
        for head_kwargs in task_heads_kwargs_list:
            self.task_heads[head_kwargs["task_name"]] = TaskHead(in_dim=self.in_dim, **head_kwargs)

    # Return a dictionary: Dict[task_name, Tensor]
    def forward(self, h: torch.Tensor):
        task_head_outputs = {}

        for task, head in self.task_heads.items():
            task_head_outputs[task] = head.forward(h)

        return task_head_outputs

    def __repr__(self):
        task_repr = []
        for head in self.task_heads:
            task_repr.append(head.__repr__())

    @property
    def out_dim(self) -> Dict[str, int]:
        r"""
        Returns the output dimension of each task head
        """
        return {task_name: head.out_dim for task_name, head in self.task_heads.items()}


class FullGraphMultiTaskNetwork(FullGraphNetwork):
    """
    Class that allows to implement a full multi-task graph neural network architecture,
    including the pre-processing MLP, post-processing MLP and the task-specific heads.

    In this model, the tasks share a full DGL network as a "trunk", and then they have task-specific MLPs.

    Each molecular graph is associated with a variety of tasks, so the network should output the task-specific preedictions for a graph.
    """

    def __init__(
        self,
        task_heads_kwargs_list: List[Dict[str, Any]],
        gnn_kwargs: Dict[str, Any],
        pre_nn_kwargs: Optional[Dict[str, Any]] = None,
        pe_encoders_kwargs: Optional[Dict[str, Any]] = None,
        pre_nn_edges_kwargs: Optional[Dict[str, Any]] = None,
        post_nn_kwargs: Optional[Dict[str, Any]] = None,
        num_inference_to_average: int = 1,
        name: str = "Multitask_GNN",
    ):
        r"""
        Class that allows to implement a full multi-task graph neural network architecture,
        including the pre-processing MLP, post-processing MLP and the task-specific heads.

        In this model, the tasks share a full DGL network as a "trunk", and additionally have task-specific MLPs.
        Each molecular graph is associated with a variety of tasks, so the network outputs the task-specific preedictions for a graph.

        Parameters:

            task_heads_kwargs_list:
                This argument is a list of dictionaries containing the arguments for task heads. Each argument is used to
                initialize a task-specific MLP.

            gnn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                GNN network using the class `FeedForwardDGL`.
                It must respect the following criteria:

                - gnn_kwargs["in_dim"] must be equal to pre_nn_kwargs["out_dim"]
                - gnn_kwargs["out_dim"] must be equal to post_nn_kwargs["in_dim"]

            pre_nn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the node features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            pre_nn_edges_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the edge features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            post_nn_kwargs:
                key-word arguments to use for the initialization of the post-processing
                MLP network after the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a post-processing MLP.

            num_inference_to_average:
                Number of inferences to average at val/test time. This is used to avoid the noise introduced
                by positional encodings with sign-flips. In case no such encoding is given,
                this parameter is ignored.
                NOTE: The inference time will be slowed-down proportionaly to this parameter.

            name:
                Name attributed to the current network, for display and printing
                purposes.
        """

        super().__init__(
            gnn_kwargs=gnn_kwargs,
            pre_nn_kwargs=pre_nn_kwargs,
            pre_nn_edges_kwargs=pre_nn_edges_kwargs,
            pe_encoders_kwargs=pe_encoders_kwargs,
            post_nn_kwargs=post_nn_kwargs,
            num_inference_to_average=num_inference_to_average,
            name=name,
        )

        self.task_heads = TaskHeads(in_dim=super().out_dim, task_heads_kwargs_list=task_heads_kwargs_list)

    def forward(self, g: Union[DGLGraph, Data]):
        h = super().forward(g)
        return self.task_heads(h)

    @property
    def out_dim(self) -> Dict[str, int]:
        r"""
        Returns the output dimension of the network for each task
        """
        return self.task_heads.out_dim

    def __repr__(self):
        return super().__repr__()
