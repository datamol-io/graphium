from torch import nn
import torch.nn.functional as F
import torch
import math
from copy import deepcopy
import dgl
from dgl.nn.pytorch.glob import mean_nodes, sum_nodes
from typing import List, Dict

from goli.dgl.base_layers import FCLayer, get_activation
from goli.dgl.dgl_layers.pooling import parse_pooling_layer, VirtualNode

from goli.dgl.dgl_layers import (
    GATLayer,
    GCNLayer,
    GINLayer,
    GatedGCNLayer,
    PNAConvolutionalLayer,
    PNAMessagePassingLayer,
)

from goli.dgl.residual_connections import (
    ResidualConnectionConcat,
    ResidualConnectionDenseNet,
    ResidualConnectionNone,
    ResidualConnectionSimple,
    ResidualConnectionWeighted,
)


LAYERS_DICT = {
    "fc": FCLayer,
    "gcn": GCNLayer,
    "gin": GINLayer,
    "gat": GATLayer,
    "gated-gcn": GatedGCNLayer,
    "pna-conv": PNAConvolutionalLayer,
    "pna-msgpass": PNAMessagePassingLayer,
}


RESIDUALS_DICT = {
    "none": ResidualConnectionNone,
    "simple": ResidualConnectionSimple,
    "weighted": ResidualConnectionWeighted,
    "concat": ResidualConnectionConcat,
    "densenet": ResidualConnectionDenseNet,
}


class FeedForwardNN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims,
        activation="relu",
        last_activation="none",
        batch_norm=False,
        dropout=0.25,
        residual_type="none",
        residual_skip_steps=1,
        name="LNN",
        layer_type=FCLayer,
        **layer_kwargs,
    ):

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
        self.residual_type = residual_type
        self.residual_skip_steps = residual_skip_steps
        self.layer_kwargs = layer_kwargs
        self.name = name

        # Parse the layer and residuals
        self.layer_class, self.layer_name = self._parse_class_from_dict(layer_type, LAYERS_DICT)
        self.residual_class, self.residual_name = self._parse_residual(residual_type, RESIDUALS_DICT)

        self._register_hparams()

        self.full_dims = [self.in_dim] + self.hidden_dims + [self.out_dim]
        self._create_layers()

    def _register_hparams(self):
        # Register the Hyper-parameters to be compatible with Pytorch-Lightning and Tensorboard
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

    def _parse_class_from_dict(self, name_or_class, class_dict):
        if isinstance(name_or_class, str):
            obj_name = name_or_class.lower()
            obj_class = class_dict[obj_name]
        elif isinstance(callable):
            obj_name = str(name_or_class)
            obj_class = name_or_class
        else:
            raise TypeError(f"`name_or_class` must be str or callable, provided: {type(name_or_class)}")

        return obj_class, obj_name

    def _create_residual_connection(self):

        if self.residual_class.has_weights():
            self.residual_layer = self.residual_class(
                skip_steps=self.residual_skip_steps,
                full_dims=self.full_dims,
                dropout=self.dropout,
                activation=self.activation,
                batch_norm=self.batch_norm,
                bias=self.bias,
            )
        else:
            self.residual_layer = self.residual_class(skip_steps=self.residual_skip_steps)

        residual_out_dims = self.residual_layer.get_true_out_dims(self.full_dims[1:])

        return residual_out_dims

    def _create_layers(self):

        residual_out_dims = self._create_residual_connection()

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

            this_in_dim = residual_out_dims[ii]

        self.out_linear = nn.Linear(in_features=this_in_dim, out_features=self.out_dim)
        

    def forward(self, h):
        h_prev = None
        for ii, layer in enumerate(self.layers):
            h = layer.forward(h)
            h, h_prev = self.residual.forward(h, h_prev, step_idx=ii)

        h = self.out_linear.forward(h)

        return h

    def __repr__(self):
        r"""
        Controls how the class is printed
        """
        class_str = f"{self.__class__.__name__}(depth={self.depth})"
        layer_str = f"[{self.layer_name}[{' -> '.join(map(str, self.full_dims))}]"
        out_str = " -> Linear({self.out_dim})"

        return class_str + layer_str + out_str


class FeedForwardDGL(FeedForwardNN):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims,
        activation="relu",
        last_activation="none",
        batch_norm=False,
        dropout=0.25,
        residual_type="none",
        residual_skip_steps=1,
        edge_dim=0,
        pooling="sum",
        name="GNN",
        layer_class="gcn",
        virtual_node="none",
        **layer_kwargs,
    ):

        # Initialize the additional attributes
        self.edge_dim = edge_dim
        self.edge_features = self.edge_dim > 0
        self.pooling = pooling.lower()
        self.virtual_node = virtual_node.lower()

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
            layer_class=layer_class,
            dropout=dropout,
            **layer_kwargs,
        )

        # Initialize global and virtual node layers
        self.global_pool_layer, out_pool_dim = parse_pooling_layer(out_dim, self.pooling)
        self._initialize_virtual_node_layers()

        # Initialize input and output linear layers
        self.in_linear = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dims[0])

    def _register_hparams(self):
        return super()._register_hparams()
        self.hparams["edge_dim"] = self.edge_dim
        self.hparams["pooling"] = self.pooling
        self.hparams["intermittent_pooling"] = self.intermittent_pooling

    def _create_layers(self):

        residual_out_dims = self._create_residual_connection()

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

            # Get the true input dimension of the next layer,
            # by factoring both the residual connection and GNN layer type
            this_in_dim = residual_out_dims[ii] * layers[ii-1].out_dim_factor

        self.out_linear = nn.Linear(in_features=this_in_dim, out_features=self.out_dim)
        

    def _initialize_virtual_node_layers(self):
        self.virtual_node_layers = nn.ModuleList()
        in_dims = self.full_dims[:-1]
        out_dims = self.full_dims[1:]

        for ii in range(self.depth):

            # Compute the input and output dims depending on the residual type
            in_dim, out_dim = self._get_residual_in_out_dim(step_idx=ii)

            self.virtual_node_layers.append(
                VirtualNode(
                    dim=out_dim,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    bias=True,
                    vn_type=self.virtual_node,
                    residual=self.residual,
                )
            )

    def _pool_layer_forward(self, graph, h):

        # Pool the nodes together
        if len(self.global_pool_layer) > 0:
            pooled_h = []
            for this_pool in self.global_pool_layer:
                pooled_h.append(this_pool(graph, h))
            pooled_h = torch.cat(pooled_h, dim=-1)
        else:
            pooled_h = h

        pooled_h = self.out_linear(pooled_h)

        return pooled_h

    def _layer_forward(self, layer, graph, h, e):
        if layer.layer_inputs_edges and layer.layer_outputs_edges:
            h, e = layer(g=graph, h=h, e=e)
        elif layer.layer_inputs_edges:
            h = layer(g=graph, h=h, e=e)
        elif layer.layer_outputs_edges:
            h, e = layer(g=graph, h=h)
        else:
            h = layer(g=graph, h=h)

        return h, e

    def _virtual_node_forward(self, g, h, vn_h, step_idx):
        if step_idx == 0:
            vn_h = 0
        if step_idx < len(self.virtual_node_layers):
            vn_h, h = self.virtual_node_layers[step_idx].forward(g, h, vn_h)

        return vn_h, h

    def forward(self, graph):

        # Get graph features and apply linear layer
        h = graph.ndata["h"]
        e = graph.edata["e"] if self.edge_features else None
        h = self.in_linear(h)

        h_prev = None
        e_prev = None
        vn_h = 0
        for ii, layer in enumerate(self.layers):
            h, e = self._layer_forward(layer=layer, graph=graph, h=h, e=e)
            h, h_prev = self.residual.forward(h, h_prev, step_idx=ii)
            # TODO: ADD RESIDUAL CONNECTION FOR EDGES IF APPLICABLE
            vn_h, h = self._virtual_node_forward(graph, h, vn_h, step_idx=ii)

        pooled_h = self._pool_layer_forward(graph, h)

        return pooled_h


class FullDGLNetwork(nn.Module):
    def __init__(
        self,
        pre_nn_kwargs,
        gnn_kwargs,
        post_nn_kwargs,
        name="DGL_GNN",
    ):

        # Initialize the parent nn.Module
        super().__init__()
        self.name = name

        self.pre_nn, self.post_nn = None, None
        if pre_nn_kwargs is not None:
            self.pre_nn = FeedForwardNN(**pre_nn_kwargs, name="NN-pre-trans")
        self.gnn = FeedForwardDGL(**gnn_kwargs, name="main-GNN")
        if post_nn_kwargs is not None:
            self.post_nn = FeedForwardNN(**post_nn_kwargs, name="NN-post-trans")

        hparams_temp = {**self.pre_nn.hparams, **self.pre_nn.hparams, **self.pre_nn.hparams}
        self.hparams = {f"{self.name}.{key}": elem for key, elem in hparams_temp.items()}

    def forward(self, graph):
        if self.pre_nn is not None:
            h = graph.ndata["h"]
            h = self.pre_nn.forward(h)
            graph.ndata["h"] = h
        h = self.gnn.forward(graph)
        if self.post_nn is not None:
            h = self.post_nn.forward(h)
        return h


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
