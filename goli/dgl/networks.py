from torch import nn
import torch.nn.functional as F
import torch
import math
from copy import deepcopy
import dgl

from goli.dgl.dgl_layers.gcn_layer import GCNLayer
from goli.dgl.dgl_layers.gin_layer import GINLayer
from goli.dgl.dgl_layers.gat_layer import GATLayer
from goli.dgl.dgl_layers.gated_gcn_layer import GatedGCNLayer
from goli.dgl.dgl_layers.pna_layer import PNALayer, PNASimpleLayer
from goli.dgl.dgl_layers.intermittent_pooling_layer import IntermittentPoolingLayer

from goli.dgl.base_layers import FCLayer, get_activation, parse_pooling_layer


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
        name="LNN",
        layer_type=FCLayer,
        **kwargs,
    ):

        super().__init__()

        # Hyper-parameters
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = list(hidden_dims)
        self.depth = len(hidden_dims) + 1
        self.activation = get_activation(activation)
        self.last_activation = get_activation(last_activation)
        self.dropout = dropout
        self.layer_type = layer_type
        self.batch_norm = batch_norm
        self.kwargs = kwargs
        self.name = name

        self.hparams = {
            f"{self.name}.out_dim": self.out_dim,
            f"{self.name}.hidden_dims": self.hidden_dims,
            f"{self.name}.activation": str(self.activation),
            f"{self.name}.last_activation": str(self.last_activation),
            f"{self.name}.layer_type": str(layer_type),
            f"{self.name}.depth": self.depth,
            f"{self.name}.dropout": self.dropout,
        }

        self.full_dims = [self.in_dim] + self.hidden_dims + [self.out_dim]
        self.true_in_dims, self.true_out_dims = self.create_layers(
            in_dims=self.full_dims[:-1], out_dims=self.full_dims[1:]
        )

    def create_layers(self, in_dims: list, out_dims: list, kwargs_of_lists=None):

        # Create the layers
        self.layers = nn.ModuleList()
        for ii in range(self.depth):
            this_activation = self.activation
            if ii == self.depth - 1:
                this_activation = self.last_activation

            if kwargs_of_lists is None:
                this_kwargs = {}
            else:
                this_kwargs = {key: value[ii] for key, value in kwargs_of_lists.items()}

            self.layers.append(
                self.layer_type(
                    in_dim=in_dims[ii],
                    out_dim=out_dims[ii],
                    activation=this_activation,
                    dropout=self.dropout,
                    batch_norm=self.batch_norm,
                    **self.kwargs,
                    **this_kwargs,
                )
            )

        return in_dims, out_dims

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


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
        edge_features=False,
        pooling="sum",
        name="GNN",
        layer_type="gcn",
        intermittent_pooling="none",
        **kwargs,
    ):

        layer_type, layer_name = self._parse_gnn_layer(layer_type)

        # Initialize the parent `FeedForwardNN`
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            last_activation=last_activation,
            batch_norm=batch_norm,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            **kwargs,
        )

        # Initialize the additional attributes
        self.edge_features = edge_features
        self.pooling = pooling.lower()
        self.intermittent_pooling = intermittent_pooling.lower()

        # Register some hparams for cross-validation
        self.hparams["layer_name"] = layer_name
        self.hparams["layer_type"] = layer_type
        self.hparams["edge_features"] = edge_features
        self.hparams["pooling"] = pooling
        self.hparams["intermittent_pooling"] = intermittent_pooling
        self.hparams["activation"] = str(activation)
        self.hparams["last_activation"] = str(last_activation)
        self.hparams["batch_norm"] = str(batch_norm)
        self.hparams["dropout"] = str(dropout)
        self.hparams["layer_kwargs"] = str(kwargs)

        # Initialize global and intermittent pooling layers
        self.global_pool_layer, out_pool_dim = parse_pooling_layer(out_dim, self.pooling)
        self.intermittent_pool_layers = nn.ModuleList(
            [
                IntermittentPoolingLayer(
                    in_dim=this_dim,
                    num_layers=2,
                    pooling=self.intermittent_pooling,
                    activation=activation,
                    last_activation=last_activation,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    bias=True,
                )
                for ii, this_dim in enumerate(self.true_out_dims[1:])
            ]
        )

        # Initialize input and output linear layers
        self.in_linear = nn.Linear(in_features=self.in_dim, out_features=self.hidden_dims[0])
        self.out_linear = nn.Linear(in_features=out_pool_dim, out_features=self.hidden_dims[-1])

    def _parse_gnn_layer(self, layer_type):
        # Parse the GNN type from the name
        if isinstance(layer_type, str):
            layer_name = layer_type.lower()
            if layer_name == "gcn":
                layer_type = GCNLayer
            elif layer_name == "gin":
                layer_type = GINLayer
            elif layer_name == "gat":
                layer_type = GATLayer
            elif layer_name == "gated-gcn":
                layer_type = GatedGCNLayer
            elif layer_name == "pna":
                layer_type = PNALayer
            elif layer_name == "pna-simple":
                layer_type = PNASimpleLayer
            else:
                raise ValueError(f"Unsupported `layer_type`: {layer_type}")
        else:
            layer_name = str(layer_type)

        return layer_type, layer_name

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

    def create_layers(self, in_dims: list, out_dims: list):

        in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove = self._get_layers_args()
        for key in kwargs_keys_to_remove:
            self.kwargs.pop(key)
        super().create_layers(in_dims=in_dims, out_dims=out_dims, kwargs_of_lists=kwargs_of_lists)
        return in_dims, true_out_dims

    def _forward_pre_layers(self, graph):
        h = graph.ndata["hv"]
        e = graph.edata["he"] if self.edge_features else None
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

    def _forward_intermittent_pool_layers(self, graph, h, layer_idx):
        idx = layer_idx - 1
        if (
            (len(self.intermittent_pool_layers) > 0)
            and (idx >= 0)
            and (idx < len(self.intermittent_pool_layers))
        ):

            h = self.intermittent_pool_layers[idx].forward(graph, h)

        return h

    def forward(self, graph):

        h, e = self._forward_pre_layers(graph)

        # Apply the consecutive graph convolutions
        for ii, cv_layer in enumerate(self.layers):
            h, e = self._forward_middle_layers(cv_layer, graph, h, e)
            h = self._forward_intermittent_pool_layers(graph, h, ii)

        pooled_h = self._forward_post_layers(graph, h)

        return pooled_h


class SkipFeedForwardDGL(FeedForwardDGL):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims,
        activation="relu",
        last_activation="none",
        batch_norm=False,
        dropout=0.25,
        edge_features=False,
        pooling="sum",
        skip_steps=2,
        name="GNN",
        layer_type="gcn",
        intermittent_pooling="none",
        **kwargs,
    ):

        assert isinstance(skip_steps, int)
        self.skip_steps = skip_steps

        # Initialize the parent `FeedForwardDGL`
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            last_activation=last_activation,
            batch_norm=batch_norm,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            edge_features=edge_features,
            pooling=pooling,
            intermittent_pooling=intermittent_pooling,
            **kwargs,
        )

        self.hparams["skip_steps"] = skip_steps

    def _get_layers_args(self):
        in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove = super()._get_layers_args()
        for ii in range(1, len(in_dims)):
            if (self.skip_steps != 0) and ((ii % self.skip_steps) == 0):
                in_dims[ii] *= 2
                if "in_dim_e" in kwargs_of_lists.keys():
                    kwargs_of_lists["in_dim_e"][ii] *= 2

        return in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove

    def forward(self, graph):

        h, e = self._forward_pre_layers(graph)

        # Apply the consecutive graph convolutions
        for ii, cv_layer in enumerate(self.layers):
            # Concatenate with h_prev and e_prev
            if (self.skip_steps != 0) and ((ii % self.skip_steps) == 0) and (ii > 0):
                h = torch.cat([h, h_prev], dim=-1)
                if e is not None:
                    e = torch.cat([e, e_prev], dim=-1)

            h, e = self._forward_middle_layers(cv_layer, graph, h, e)
            h = self._forward_intermittent_pool_layers(graph, h, ii)

            if (self.skip_steps != 0) and ((ii % self.skip_steps) == 0):
                h_prev = h
                e_prev = e

        pooled_h = self._forward_post_layers(graph, h)

        return pooled_h


class DenseNetDGL(FeedForwardDGL):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims,
        activation="relu",
        last_activation="none",
        batch_norm=False,
        dropout=0.25,
        edge_features=False,
        pooling="sum",
        skip_steps=1,
        name="GNN",
        layer_type="gcn",
        intermittent_pooling="none",
        **kwargs,
    ):

        assert isinstance(skip_steps, int)
        self.skip_steps = skip_steps

        # Initialize the parent `FeedForwardDGL`
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            last_activation=last_activation,
            batch_norm=batch_norm,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            edge_features=edge_features,
            pooling=pooling,
            intermittent_pooling=intermittent_pooling,
            **kwargs,
        )

        self.hparams["skip_steps"] = skip_steps

    def _get_layers_args(self):
        in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove = super()._get_layers_args()
        skip_counter = 2
        for ii in range(1, len(in_dims)):
            if (ii % self.skip_steps) == 0:
                in_dims[ii] *= skip_counter
                if "in_dim_e" in kwargs_of_lists.keys():
                    kwargs_of_lists["in_dim_e"][ii] *= skip_counter
                skip_counter += 1

        return in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove

    def forward(self, graph):

        h, e = self._forward_pre_layers(graph)

        # Apply the consecutive graph convolutions
        for ii, cv_layer in enumerate(self.layers):
            # Concatenate with h_prev and e_prev
            if (self.skip_steps != 0) and ((ii % self.skip_steps) == 0) and (ii > 0):
                h = torch.cat([h, h_prev], dim=-1)
                if e is not None:
                    e = torch.cat([e, e_prev], dim=-1)

            if (self.skip_steps != 0) and ((ii % self.skip_steps) == 0) and (ii >= 0):
                h_prev = h
                e_prev = e

            h, e = self._forward_middle_layers(cv_layer, graph, h, e)
            h = self._forward_intermittent_pool_layers(graph, h, ii)

        pooled_h = self._forward_post_layers(graph, h)

        return pooled_h


class ResNetDGL(FeedForwardDGL):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dims,
        activation="relu",
        last_activation="none",
        batch_norm=False,
        dropout=0.25,
        edge_features=False,
        pooling="sum",
        skip_steps=2,
        residual_weights=False,
        name="GNN",
        layer_type="gcn",
        intermittent_pooling="none",
        **kwargs,
    ):

        assert isinstance(skip_steps, int)
        self.skip_steps = skip_steps
        self.residual_weights = residual_weights

        # Initialize the parent `FeedForwardDGL`
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            last_activation=last_activation,
            batch_norm=batch_norm,
            name=name,
            layer_type=layer_type,
            dropout=dropout,
            edge_features=edge_features,
            pooling=pooling,
            intermittent_pooling=intermittent_pooling,
            **kwargs,
        )

        self.hparams["skip_steps"] = skip_steps
        self.hparams["residual_weights"] = residual_weights

        self.h_residual_list = self._make_residual_weigts(residual_weights)

    def _make_residual_weigts(self, residual_weights: bool):

        residual_list = None
        if residual_weights:
            residual_list = nn.ModuleList()
            for ii in range(1, self.depth, self.skip_steps):
                this_dim = self.full_dims[ii]
                residual_list.append(
                    nn.Sequential(
                        FCLayer(
                            this_dim,
                            this_dim,
                            activation=self.activation,
                            dropout=self.dropout,
                            batch_norm=self.batch_norm,
                            bias=False,
                        ),
                        nn.Linear(this_dim, this_dim, bias=False),
                    )
                )
        return residual_list

    def forward(self, graph):

        h, e = self._forward_pre_layers(graph)
        skip_count = 0

        # Apply the consecutive graph convolutions
        for ii, cv_layer in enumerate(self.layers):
            # Concatenate with h_prev and e_prev
            if (self.skip_steps != 0) and ((ii % self.skip_steps) == 1):
                if ii > 1:
                    h = h + h_prev
                    if e is not None:
                        e = e + e_prev

                if self.residual_weights:
                    h_prev = self.h_residual_list[skip_count].forward(h)
                else:
                    h_prev = h

                e_prev = e

                skip_count += 1

            h, e = self._forward_middle_layers(cv_layer, graph, h, e)
            h = self._forward_intermittent_pool_layers(graph, h, ii)

        pooled_h = self._forward_post_layers(graph, h)

        return pooled_h


class DGLGraphNetwork(nn.Module):
    def __init__(
        self,
        gnn_kwargs,
        lnn_kwargs=None,
        gnn_architecture="skip-concat",
        lnn_architecture=None,
        name="DGL_GNN",
    ):

        # Initialize the parent nn.Module
        super().__init__()
        self.name = name
        self.gnn_architecture = gnn_architecture
        self.lnn_architecture = lnn_architecture

        self.gnn = self._parse_gnn_architecture(gnn_architecture, gnn_kwargs)
        self.lnn = self._parse_lnn_architecture(lnn_architecture, lnn_kwargs)

        self.hparams = {f"{self.name}.{key}": elem for key, elem in self.gnn.hparams.items()}
        self.hparams[f"{self.name}.gnn_architecture"] = self.gnn_architecture
        self.hparams[f"{self.name}.lnn_architecture"] = self.lnn_architecture

    def _parse_gnn_architecture(self, gnn_architecture, gnn_kwargs):
        gnn_architecture = gnn_architecture.lower()
        if gnn_architecture == "skip-concat":
            gnn = SkipFeedForwardDGL(**gnn_kwargs)
        elif gnn_architecture == "resnet":
            gnn = ResNetDGL(**gnn_kwargs)
        elif gnn_architecture == "resnet-residualweights":
            gnn_kwargs["residual_weights"] = True
            gnn = ResNetDGL(**gnn_kwargs)
        elif gnn_architecture == "densenet":
            gnn = DenseNetDGL(**gnn_kwargs)
        else:
            raise NotImplementedError

        return gnn

    def _parse_lnn_architecture(self, lnn_architecture, lnn_kwargs):
        if lnn_architecture is not None:
            raise NotImplementedError

        if lnn_kwargs is None:
            lnn = None
        else:
            lnn = FeedForwardNN(**lnn_kwargs)

        return lnn

    def forward(self, graph):

        h = self.gnn.forward(graph)
        if self.lnn is not None:
            h = self.lnn.forward(h)
        return h


class SiameseGraphNetwork(DGLGraphNetwork):
    def __init__(self, gnn_kwargs, dist_method, gnn_architecture="skip-concat", name="SiameseGNN"):

        # Initialize the parent nn.Module
        super().__init__(
            gnn_kwargs=gnn_kwargs,
            lnn_kwargs=None,
            gnn_architecture=gnn_architecture,
            lnn_architecture=None,
            name=name,
        )

        self.dist_method = dist_method.lower()
        self.activation = "sigmoid"

        self.hparams[f"{self.name}.dist_method"] = self.dist_method

    def forward(self, graphs):
        graph_1, graph_2 = graphs

        out_1 = self.gnn.forward(graph_1)
        out_2 = self.gnn.forward(graph_2)

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
