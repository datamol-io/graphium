import torch
from typing import Callable, Union, Optional
from functools import partial

import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

from goli.nn.base_graph_layer import BaseGraphModule, check_intpus_allow_int
from goli.nn.base_layers import MLP
from goli.utils.decorators import classproperty


class MPNNPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: Optional[int] = None,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        use_edges=False,
    ):
        r"""
            MPNNPyg: InteractionNetwork layer witg edges and global feature, GPS++ type of GNN layer
            GPS++: An Optimised Hybrid MPNN/Transformer for Molecular Property Prediction
            Dominic Masters, Josef Dean, Kerstin Klaser, Zhiyi Li, Sam Maddrell-Mander, Adam Sanders,
            Hatem Helal, Deniz Beker, Ladislav Rampášek, Dominique Beaini
            https://arxiv.org/abs/2212.02229

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

            init_eps :
                Initial :math:`\epsilon` value, default: ``0``.

            learn_eps :
                If True, :math:`\epsilon` will be a learnable parameter.

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
        )

        self.use_edges = use_edges
        self.gather_from = gather_from
        self.node_combine_method = node_combine_method

        gin_nn = MLP(
            in_dim=self.in_dim,
            hidden_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=2,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )

        self.node_model = MLP(
            in_dim=self.in_dim,
            hidden_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=2,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )

        self.edge_model = MLP(
            in_dim=self.in_dim,
            hidden_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=2,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )

        self.model = pyg_nn.GINEConv(gin_nn, edge_dim=in_dim_edges)  # , node_dim=-1)
        self.model.__check_input__ = partial(check_intpus_allow_int, self)

    def gather_features(self, input_features, senders, receivers):
        out = []

        receiver_features = input_features[receivers]
        sender_features = input_features[senders]

        if self.gather_from == 'receivers':
            out.append(receiver_features)

        if self.gather_from == 'senders':
            out.append(sender_features)

        if self.gather_from == 'both':
            if self.node_combine_method == 'sum':
                out.append(receiver_features + sender_features)
            elif self.node_combine_method == 'concat':
                torch.cat([receiver_features, sender_features], dim=-1)
            else:
                raise ValueError(f"node_combine_method {self.node_combine_method} not recognised.")

        return out, sender_features, receiver_features

    def forward(self, batch):
        nodes_input = batch.h
        edges_input = batch.edge_attr
        senders = batch.edge_index[0]
        receivers = batch.edge_index[1]

        # ---------------EDGE step---------------
        edge_model_input, sender_nodes, receiver_nodes = self.gather_features(nodes_input, senders, receivers)

        if self.use_edges:
            edge_model_input.append(edges_input)
            edge_model_input = torch.cat([edge_model_input[0], edge_model_input[1]], dim=-1)

            edges = self.edge_model(edge_model_input)
            if 'before_scatter' in self.edge_dropout_loc:
                edges = self.edge_dropout(edges, training=training)
        else:
            edges = edge_model_input

        # ---------------NODE step---------------
        if self.use_edges:
            sender_nodes = self.edge_dna_dropout['senders'](sender_nodes, training=training)
            receiver_nodes = self.edge_dna_dropout['receivers'](receiver_nodes, training=training)
        # message + aggregate
        node_model_input = self.aggregate_features(edges, senders, receivers, sender_nodes, receiver_nodes,
                                                   self.n_nodes_per_pack)
        node_model_input.append(nodes_input)
        nodes = torch.cat(node_model_input, axis=-1)
        nodes = self.node_model(nodes)

        # ---------------- Stochastic depth  ---------------
        '''
        nodes = self.graph_dropout(nodes, node_graph_idx)
        if self.use_edges:
            edges = self.graph_dropout(edges, edge_graph_idx)
        if self.use_globals:
            global_latent = self.graph_dropout(global_latent, None)

        # dropout before the residual block`
        nodes = self.node_dropout(nodes, training=training)
        nodes = self.residual_add(nodes_input, nodes)
        if self.output_scale != 1.0:
            nodes *= tf.cast(self.output_scale, nodes.dtype)

        if self.use_edges:
            if self.edge_dropout_loc == 'before_residual_add':
                edges = self.edge_dropout(edges, training=training)
            edges = self.residual_add(edges_input, edges)
            if self.output_scale != 1.0:
                edges *= tf.cast(self.output_scale, edges.dtype)
        '''
        batch.h = nodes
        batch.edge_attr = edges
        # batch.h = self.model(batch.h, batch.edge_index, batch.edge_attr)
        # batch.h = self.apply_norm_activation_dropout(batch.h)

        return batch

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            supports_edges: bool
                Always ``True`` for the current class
        """
        return True

    @property
    def layer_inputs_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``True`` for the current class
        """
        return True

    @property
    def layer_outputs_edges(self) -> bool:
        r"""
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``False`` for the current class
        """
        return False

    @property
    def out_dim_factor(self) -> int:
        r"""
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns:

            int:
                Always ``1`` for the current class
        """
        return 1
