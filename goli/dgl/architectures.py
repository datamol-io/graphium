from torch import nn
import torch.nn.functional as F
import torch
import math
from copy import deepcopy
import dgl
from dgl.nn.pytorch.glob import mean_nodes, sum_nodes
from typing import List, Dict

from goli.dgl.dgl_layers.gcn_layer import GCNLayer
from goli.dgl.dgl_layers.gin_layer import GINLayer
from goli.dgl.dgl_layers.gat_layer import GATLayer
from goli.dgl.dgl_layers.gated_gcn_layer import GatedGCNLayer
from goli.dgl.dgl_layers.pna_layer import PNAComplexLayer, PNASimpleLayer
from goli.dgl.dgl_layers.intermittent_pooling_layer import IntermittentPoolingLayer

from goli.dgl.base_layers import FCLayer, get_activation, parse_pooling_layer, VirtualNode
from goli.dgl.residual_connections import RESIDUAL_TYPE_DICT


LAYERS_DICT = {
    "fc": FCLayer,
    "gcn": GCNLayer,
    "gin": GINLayer,
    "gat": GATLayer,
    "gated-gcn": GatedGCNLayer,
    "pna-complex": PNAComplexLayer,
    "pna-simple": PNASimpleLayer,
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
        **kwargs,
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
        self.layer_type = layer_type
        self.batch_norm = batch_norm
        self.residual_type = residual_type
        self.residual_skip_steps = residual_skip_steps
        self.kwargs = kwargs
        self.name = name
        self.layer_type, self.layer_name = self._parse_layer(layer_type)

        self._register_hparams()

        self.full_dims = [self.in_dim] + self.hidden_dims + [self.out_dim]
        self.true_in_dims, self.true_out_dims = self._create_layers(
            in_dims=self.full_dims[:-1], out_dims=self.full_dims[1:]
        )

    def _register_hparams(self):
        # Register the Hyper-parameters to be compatible with Pytorch-Lightning and Tensorboard
        self.hparams = {
            f"{self.name}.out_dim": self.out_dim,
            f"{self.name}.hidden_dims": self.hidden_dims,
            f"{self.name}.activation": str(self.activation),
            f"{self.name}.batch_norm": str(self.batch_norm),
            f"{self.name}.last_activation": str(self.last_activation),
            f"{self.name}.layer_type": str(layer_type),
            f"{self.name}.depth": self.depth,
            f"{self.name}.dropout": self.dropout,
            f"{self.name}.residual_type": self.residual_type,
            f"{self.name}.residual_skip_steps": self.residual_skip_steps,
            f"{self.name}.layer_kwargs": str(self.layer_kwargs),
        }

    def _parse_layer(self, layer_type):
        # Parse the GNN type from the name
        if isinstance(layer_type, str):
            layer_name = layer_type.lower()
            layer_type = LAYERS_DICT[layer_name]
        else:
            layer_name = str(layer_type)

        return layer_type, layer_name

    def _get_residual_in_out_dim(self, step_idx):
        in_dims = self.full_dims[:-1]
        out_dims = self.full_dims[1:]
        cum_out_dims = torch.cumsum(torch.tensor(out_dims))

        # Compute the input and output dims depending on the residual type
        in_dim = in_dims[step_idx]
        out_dim = out_dims[step_idx]
        if step_idx > 0:
            if self.residual.h_dim_increase_type() == "previous":
                in_dim += out_dims[step_idx - 1]
            elif self.residual.h_dim_increase_type() == "cumulative":
                in_dim += cum_out_dims[step_idx - 1]

        return in_dim, out_dim

    def _create_layers(self, in_dims, out_dims, kwargs_of_lists=None):

        # Create the residual connections
        self.residual = self._create_residual(
            residual_type=self.residual_type,
            skip_steps=self.residual_skip_steps,
            full_dims=self.full_dims,
            dropout=self.dropout,
            activation=self.activation,
            batch_norm=self.batch_norm,
            bias=self.bias,
        )

        # Create a list for all the layers
        self.layers = nn.ModuleList()
        for ii in range(self.depth):
            this_activation = self.activation
            if ii == self.depth - 1:
                this_activation = self.last_activation

            # Get specific kwargs for the layer type
            if kwargs_of_lists is None:
                this_kwargs = {}
            else:
                this_kwargs = {key: value[ii] for key, value in kwargs_of_lists.items()}

            # Compute the input and output dims depending on the residual type
            in_dim, out_dim = self._get_residual_in_out_dim(step_idx=ii)

            # Create the layer
            self.layers.append(
                self.layer_type(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    activation=this_activation,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    **self.kwargs,
                    **this_kwargs,
                )
            )

        return in_dims, out_dims

    def _create_residual(self, residual_type, skip_steps, **kwargs):
        residual_class = RESIDUAL_TYPE_DICT[residual_type]
        if residual_class.has_weights():
            residual = residual_class(skip_steps=skip_steps, **kwargs)
        else:
            residual = residual_class(skip_steps=skip_steps)
        return residual

    def forward(self, h):
        h_prev = None
        for ii, layer in enumerate(self.layers):
            h = layer.forward(h)
            h, h_prev = self.residual.forward(h, h_prev, step_idx=ii)

        return h


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
        edge_features=False,
        pooling="sum",
        name="GNN",
        layer_type="gcn",
        virtual_node="none",
        **kwargs,
    ):

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
            **kwargs,
        )

        # Initialize the additional attributes
        self.edge_features = edge_features
        self.pooling = pooling.lower()
        self.virtual_node = virtual_node.lower()

        # Register some hparams for cross-validation
        self._register_hparams()

        # Initialize global and virtual node layers
        self.global_pool_layer, out_pool_dim = parse_pooling_layer(out_dim, self.pooling)
        self._initialize_virtual_node_layers()

        # Initialize input and output linear layers
        self.in_linear = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dims[0])
        self.out_linear = nn.Linear(in_features=out_pool_dim, out_features=self.hidden_dims[-1])

    def _register_hparams(self):
        return super()._register_hparams()
        self.hparams["edge_features"] = self.edge_features
        self.hparams["pooling"] = self.pooling
        self.hparams["intermittent_pooling"] = self.intermittent_pooling

    def _get_layers_args(self):
        in_dims = self.hidden_dims[0:1] + self.hidden_dims
        out_dims = self.hidden_dims + [self.out_dim]

        (
            in_dims,
            out_dims,
            true_out_dims,
            kwargs_of_lists,
            kwargs_keys_to_remove,
        ) = self.layer_type._parse_layer_args(in_dims, out_dims, **self.kwargs)

        return in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove

    def _create_layers(self, in_dims: List, out_dims: List):

        in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove = self._get_layers_args()
        for key in kwargs_keys_to_remove:
            self.kwargs.pop(key)
        super()._create_layers(in_dims=in_dims, out_dims=out_dims, kwargs_of_lists=kwargs_of_lists)
        return in_dims, true_out_dims

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

    def _forward_pre_layers(self, graph):
        h = graph.ndata["h"]
        e = graph.edata["e"] if self.edge_features else None
        h = self.in_linear(h)
        return h, e

    def _forward_post_layers(self, graph, h):

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

    def _forward_middle_layers(self, cv_layer, graph, h, e):
        if self.edge_features:
            h = cv_layer(graph, h, e)
            if isinstance(h, tuple):
                h, e = h
        else:
            h = cv_layer(graph, h)
            if isinstance(h, tuple):
                h = h[0]

        return h, e

    def _virtual_node_forward(self, g, h, vn_h, step_idx):
        if step_idx == 0:
            vn_h = 0
        if step_idx < len(self.virtual_node_layers):
            vn_h, h = self.virtual_node_layers[step_idx].forward(g, h, vn_h)

        return vn_h, h

    def forward(self, graph):

        h, e = self._forward_pre_layers(graph)
        h_prev = None
        vn_h = 0
        for ii, cv_layer in enumerate(self.layers):
            h, e = self._forward_middle_layers(cv_layer, graph, h, e)
            h, h_prev = self.residual.forward(h, h_prev, step_idx=ii)
            vn_h, h = self._virtual_node_forward(graph, h, vn_h, step_idx=ii)

        pooled_h = self._forward_post_layers(graph, h)

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
