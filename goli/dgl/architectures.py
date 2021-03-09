from torch import nn
import torch.nn.functional as F
import torch
import math
from copy import deepcopy
import dgl
from dgl.nn.pytorch.glob import mean_nodes, sum_nodes
from typing import List, Dict, Tuple, Union, Callable, Any
import inspect

from goli.dgl.base_layers import FCLayer, get_activation
from goli.dgl.dgl_layers import BaseDGLLayer
from goli.dgl.dgl_layers.pooling import parse_pooling_layer, VirtualNode
from goli.commons.spaces import FC_LAYERS_DICT, DGL_LAYERS_DICT, LAYERS_DICT


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        batch_norm: bool = False,
        residual_type: str = "none",
        residual_skip_steps: int = 1,
        name: str = "LNN",
        layer_type: Union[str, nn.Module] = "fc",
        **layer_kwargs,
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
                List of dimensions in the hidden layers.
                Be careful, the "simple" residual type only supports
                hidden dimensions of the same value.

            activation:
                activation function to use in the hidden layers.

            last_activation:
                activation function to use in the last layer.

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            batch_norm:
                Whether to use batch normalization

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

        """

        super().__init__()

        # Set the class attributes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = list(hidden_dims)
        self.depth = len(hidden_dims) + 1
        self.activation = get_activation(activation)
        self.last_activation = get_activation(last_activation)
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual_type = None if residual_type is None else residual_type.lower()
        self.residual_skip_steps = residual_skip_steps
        self.layer_kwargs = layer_kwargs
        self.name = name

        # Parse the layer and residuals
        self.layer_class, self.layer_name = self._parse_class_from_dict(layer_type, LAYERS_DICT)
        self.residual_class, self.residual_name = self._parse_class_from_dict(residual_type, RESIDUALS_DICT)

        self._register_hparams()

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

    def _register_hparams(self):
        r"""
        Register the hyperparameters for tracking by Pytorch-lightning
        """
        self.hparams = {
            f"{self.name}.out_dim": self.out_dim,
            f"{self.name}.hidden_dims": self.hidden_dims,
            f"{self.name}.activation": str(self.activation),
            f"{self.name}.batch_norm": str(self.batch_norm),
            f"{self.name}.last_activation": str(self.last_activation),
            f"{self.name}.layer_name": str(self.layer_name),
            f"{self.name}.depth": self.depth,
            f"{self.name}.dropout": self.dropout,
            f"{self.name}.residual_name": self.residual_name,
            f"{self.name}.residual_skip_steps": self.residual_skip_steps,
            f"{self.name}.layer_kwargs": str(self.layer_kwargs),
        }

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
        if self.residual_class.has_weights:
            residual_layer = self.residual_class(
                skip_steps=self.residual_skip_steps,
                out_dims=out_dims,
                dropout=self.dropout,
                activation=self.activation,
                batch_norm=self.batch_norm,
                bias=False,
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

        for ii in range(self.depth):
            this_out_dim = self.full_dims[ii + 1]
            if ii == self.depth - 1:
                this_activation = self.last_activation

            # Create the layer
            self.layers.append(
                self.layer_class(
                    in_dim=this_in_dim,
                    out_dim=this_out_dim,
                    activation=this_activation,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
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
        out_str = " -> Linear({self.out_dim})"

        return class_str + layer_str + out_str


class FeedForwardDGL(FeedForwardNN):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation: Union[str, Callable] = "relu",
        last_activation: Union[str, Callable] = "none",
        dropout: float = 0.0,
        batch_norm: bool = False,
        residual_type: str = "none",
        residual_skip_steps: int = 1,
        in_dim_edges: int = 0,
        hidden_dims_edges: List[int] = [],
        pooling: List[Union[str, Callable]] = ["sum"],
        name: str = "GNN",
        layer_type: Union[str, nn.Module] = "gcn",
        virtual_node: str = "none",
        **layer_kwargs,
    ):
        r"""
        A flexible neural network architecture, with variable hidden dimensions,
        support for multiple layer types, and support for different residual
        connections.

        This class is meant to work with different DGL-based graph neural networks
        layers. Any layer must inherit from `goli.dgl.dgl_layers.BaseDGLLayer`.

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            hidden_dims:
                List of dimensions in the hidden layers.
                Be careful, the "simple" residual type only supports
                hidden dimensions of the same value.

            activation:
                activation function to use in the hidden layers.

            last_activation:
                activation function to use in the last layer.

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            batch_norm:
                Whether to use batch normalization

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
                For graph feature predictions see `goli.dgl.dgl_layers.pooling.parse_pooling_layer`.
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
                Either a class that inherits from `goli.dgl.dgl_layers.BaseDGLLayer`,
                or one of the following strings

                - "gcn": `GCNLayer`
                - "gin": `GINLayer`
                - "gat": `GATLayer`
                - "gated-gcn": `GatedGCNLayer`
                - "pna-conv": `PNAConvolutionalLayer`
                - "pna-msgpass": `PNAMessagePassingLayer`

            virtual_node:
                A string associated to the type of virtual node to use,
                either `None`, "none", "mean", "sum", "max", "logsum".
                See `goli.dgl.dgl_layers.VirtualNode`.

                The virtual node will not use any residual connection if `residual_type`
                is "none". Otherwise, it will use a simple ResNet like residual
                connection.

        """

        # Initialize the additional attributes
        self.in_dim_edges = in_dim_edges
        self.hidden_dims_edges = hidden_dims_edges
        self.full_dims_edges = None
        if len(self.hidden_dims_edges) > 0:
            self.full_dims_edges = [self.in_dim_edges] + self.hidden_dims_edges + [self.hidden_dims_edges[-1]]

        self.virtual_node = virtual_node.lower()
        self.pooling = pooling

        # Initialize the parent `FeedForwardNN`
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            last_activation=last_activation,
            batch_norm=batch_norm,
            residual_type=residual_type,
            residual_skip_steps=residual_skip_steps,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            **layer_kwargs,
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

    def _register_hparams(self):
        r"""
        Register the hyperparameters for tracking by Pytorch-lightning
        """
        return super()._register_hparams()
        self.hparams["hidden_edge_dim"] = self.hidden_edge_dim
        self.hparams["pooling"] = self.pooling
        self.hparams["intermittent_pooling"] = self.intermittent_pooling

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
                    layer_out_dims_edges.append(self.full_dims_edges[ii + 1])
                    this_edge_kwargs["out_dim_edges"] = layer_out_dims_edges[-1]

            # Create the GNN layer
            self.layers.append(
                self.layer_class(
                    in_dim=this_in_dim,
                    out_dim=this_out_dim,
                    activation=this_activation,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    **self.layer_kwargs,
                    **this_edge_kwargs,
                )
            )

            # Create the Virtual Node layer, except at the last layer
            if ii < len(residual_out_dims):
                self.virtual_node_layers.append(
                    VirtualNode(
                        dim=this_out_dim,
                        activation=this_activation,
                        dropout=self.dropout,
                        batch_norm=self.batch_norm,
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
        self.global_pool_layer, out_pool_dim = parse_pooling_layer(layer_out_dims[-1], self.pooling)

        # Output linear layer
        self.out_linear = FCLayer(
            in_dim=out_pool_dim,
            out_dim=self.out_dim,
            activation="none",
            dropout=self.dropout,
            batch_norm=self.batch_norm,
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
            pooled_h = []
            for this_pool in self.global_pool_layer:
                pooled_h.append(this_pool(g, h))
            pooled_h = torch.cat(pooled_h, dim=-1)
        else:
            pooled_h = h

        pooled_h = self.out_linear(pooled_h)

        return pooled_h

    def _dgl_layer_forward(
        self,
        layer: BaseDGLLayer,
        g: dgl.DGLGraph,
        h: torch.Tensor,
        e: Union[torch.Tensor, None],
        h_prev: Union[torch.Tensor, None],
        e_prev: Union[torch.Tensor, None],
        step_idx: int,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        r"""
        Apply the *i-th* DGL graph layer, where *i* is the index given by `step_idx`.
        The layer is applied differently depending if there are edge features or not.

        Then, the residual is also applied on both the features and the edges (if applicable)

        Parameters:

            layer:
                The DGL layer used for the convolution

            g:
                graph on which the convolution is done

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

        # Apply the GNN layer with the right inputs/outputs
        if layer.layer_inputs_edges and layer.layer_outputs_edges:
            h, e = layer(g=g, h=h, e=e)
        elif layer.layer_inputs_edges:
            h = layer(g=g, h=h, e=e)
        elif layer.layer_outputs_edges:
            h, e = layer(g=g, h=h)
        else:
            h = layer(g=g, h=h)

        # Apply the residual layers on the features and edges (if applicable)
        if step_idx < len(self.layers) - 1:
            h, h_prev = self.residual_layer.forward(h, h_prev, step_idx=step_idx)
            if (self.residual_edges_layer is not None) and (layer.layer_outputs_edges):
                e, e_prev = self.residual_edges_layer.forward(e, e_prev, step_idx=step_idx)

        return h, e, h_prev, e_prev

    def _virtual_node_forward(
        self, g: dgl.DGLGraph, h: torch.Tensor, vn_h: torch.Tensor, step_idx: int
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

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        r"""
        Apply the full graph neural network on the input graph and node features.

        Parameters:

            g:
                graph on which the convolution is done.
                Must contain the following elements:

                - `g.ndata["h"]`: `torch.Tensor[..., N, Din]`.
                  Input node feature tensor, before the network.
                  `N` is the number of nodes, `Din` is the input features

                - `g.edata["e"]`: `torch.Tensor[..., N, Ein]` **Optional**.
                  The edge features to use. It will be ignored if the
                  model doesn't supporte edge features or if
                  `self.in_dim_edges==0`.

        Returns:

            `torch.Tensor[..., M, Dout]` or `torch.Tensor[..., N, Dout]`:
                Node or graph feature tensor, after the network.
                `N` is the number of nodes, `M` is the number of graphs,
                `Dout` is the output dimension ``self.out_dim``
                If the `self.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        # Get node and edge features
        h = g.ndata["h"]
        e = g.edata["e"] if (self.in_dim_edges > 0) else None

        # Initialize values of the residuals and virtual node
        h_prev = None
        e_prev = None
        vn_h = 0

        # Apply the forward loop of the layers, residuals and virtual nodes
        for ii, layer in enumerate(self.layers):
            h, e, h_prev, e_prev = self._dgl_layer_forward(
                layer=layer, g=g, h=h, e=e, h_prev=h_prev, e_prev=e_prev, step_idx=ii
            )
            h, vn_h = self._virtual_node_forward(g=g, h=h, vn_h=vn_h, step_idx=ii)

        pooled_h = self._pool_layer_forward(g=g, h=h)

        return pooled_h

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        class_str = f"{self.name}(depth={self.depth}, {self.residual_layer})\n    "
        layer_str = f"{self.layer_class.__name__}[{' -> '.join(map(str, self.full_dims))}]\n    "
        pool_str = f"-> Pooling({self.pooling})"
        out_str = f" -> {self.out_linear}"

        return class_str + layer_str + pool_str + out_str


class FullDGLNetwork(nn.Module):
    def __init__(
        self,
        gnn_kwargs: Dict[str, Any],
        pre_nn_kwargs: Union[type(None), Dict[str, Any]] = None,
        post_nn_kwargs: Union[type(None), Dict[str, Any]] = None,
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

            pre_nn_kwargs:
                key-word arguments to use for the initialization of the pre-processing
                MLP network of the node features before the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a pre-processing MLP.

            post_nn_kwargs:
                key-word arguments to use for the initialization of the post-processing
                MLP network after the GNN, using the class `FeedForwardNN`.
                If `None`, there won't be a post-processing MLP.

            name:
                Name attributed to the current network, for display and printing
                purposes.
        """

        super().__init__()
        self.name = name

        # Initialize the networks
        self.pre_nn, self.post_nn = None, None
        if pre_nn_kwargs is not None:
            name = pre_nn_kwargs.pop("name", "pre-trans-NN")
            self.pre_nn = FeedForwardNN(**pre_nn_kwargs, name=name)
        name = gnn_kwargs.pop("name", "main-GNN")
        self.gnn = FeedForwardDGL(**gnn_kwargs, name=name)
        if post_nn_kwargs is not None:
            name = post_nn_kwargs.pop("name", "post-trans-NN")
            self.post_nn = FeedForwardNN(**post_nn_kwargs, name=name)

        # Initialize the hyper-parameter variable to be accessible by Pytorch Lightning
        hparams_temp = {**self.pre_nn.hparams, **self.pre_nn.hparams, **self.pre_nn.hparams}
        self.hparams = {f"{self.name}.{key}": elem for key, elem in hparams_temp.items()}

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

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        r"""
        Apply the pre-processing neural network, the graph neural network,
        and the post-processing neural network on the graph features.

        Parameters:

            g:
                graph on which the convolution is done.
                Must contain the following elements:

                - `g.ndata["h"]`: `torch.Tensor[..., N, Din]`.
                  Input node feature tensor, before the network.
                  `N` is the number of nodes, `Din` is the input features dimension ``self.pre_nn.in_dim``

                - `g.edata["e"]`: `torch.Tensor[..., N, Ein]` **Optional**.
                  The edge features to use. It will be ignored if the
                  model doesn't supporte edge features or if
                  `self.in_dim_edges==0`.

        Returns:

            `torch.Tensor[..., M, Dout]` or `torch.Tensor[..., N, Dout]`:
                Node or graph feature tensor, after the network.
                `N` is the number of nodes, `M` is the number of graphs,
                `Dout` is the output dimension ``self.post_nn.out_dim``
                If the `self.gnn.pooling` is [`None`], then it returns node features and the output dimension is `N`,
                otherwise it returns graph features and the output dimension is `M`

        """

        if self.pre_nn is not None:
            h = g.ndata["h"]
            h = self.pre_nn.forward(h)
            g.ndata["h"] = h
        h = self.gnn.forward(g)
        if self.post_nn is not None:
            h = self.post_nn.forward(h)
        return h

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        pre_nn_str, post_nn_str = "", ""
        if self.pre_nn is not None:
            pre_nn_str = self.pre_nn.__repr__() + "\n\n"
        gnn_str = self.gnn.__repr__() + "\n\n"
        if self.post_nn is not None:
            post_nn_str = self.post_nn.__repr__()

        child_str = "    " + pre_nn_str + gnn_str + post_nn_str
        child_str = "    ".join(child_str.splitlines(True))

        full_str = self.name + "\n" + "-" * (len(self.name) + 2) + "\n" + child_str

        return full_str


class FullDGLSiameseNetwork(FullDGLNetwork):
    def __init__(self, pre_nn_kwargs, gnn_kwargs, post_nn_kwargs, dist_method, name="Siamese_DGL_GNN"):

        # Initialize the parent nn.Module
        super().__init__(
            pre_nn_kwargs=pre_nn_kwargs,
            gnn_kwargs=gnn_kwargs,
            post_nn_kwargs=post_nn_kwargs,
            name=name,
        )

        self.dist_method = dist_method.lower()
        self.hparams[f"{self.name}.dist_method"] = self.dist_method

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
