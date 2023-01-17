from typing import Callable, Optional, Union

import torch
from custom_ops import grouped_ops
from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.base_layers import MLP
from goli.utils.decorators import classproperty


class MPNNPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "gelu",
        dropout: float = 0.35,
        normalization: Union[str, Callable] = "layer_norm",
        gather_from: str = "both",
        scatter_to: str = "both",
        node_combine_method: str = "concat",
        num_node_mlp: int = None,
        use_edges: bool = False,
        in_dim_edges: Optional[int] = None,
        out_dim_edges: Optional[int] = None,
        num_edge_mlp: Optional[int] = None,
        edge_dropout_rate: Optional[float] = 0.0035,
    ):
        r"""
            MPNNPyg: InteractionNetwork layer witg edges and global feature, GPS++ type of GNN layer
            GPS++: An Optimised Hybrid MPNN/Transformer for Molecular Property Prediction
            Dominic Masters, Josef Dean, Kerstin Klaser, Zhiyi Li, Sam Maddrell-Mander, Adam Sanders,
            Hatem Helal, Deniz Beker, Ladislav Rampášek, Dominique Beaini
            https://arxiv.org/abs/2212.02229

        Parameters:

            in_dim:
                Input feature dimensions of the nodes

            out_dim:
                Output feature dimensions of the nodes

            activation:
                Activation function to use in the edge and node model

            dropout:
                The ratio of units to dropout at the end within apply_norm_activation_dropout.

            normalization:
                Normalization to use with the edge, node models and at the end
                within apply_norm_activation_dropout. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            gather_from:
                The method to gather features from. Could choose from:
                "senders", "receivers" and "both".

            scatter_to:
                The method to scatter features to. Could choose from:
                "senders", "receivers" and "both".

            node_combine_method:
                The method to combine the node features, Could choose from:
                "sum" and "concat".

            num_node_mlp:
                Number of mlp layer used for node model

            use_edges:
                If edge features are used

            in_dim_edges:
                Input feature dimensions of the edges

            out_dim_edges:
                Output feature dimensions of the edges

            num_edge_mlp:
                Number of mlp layer used for edge model

            edge_dropout_rate:
                dropout rate for the edges

        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
        )

        self.gather_from = gather_from
        self.scatter_to = scatter_to
        self.node_combine_method = node_combine_method
        self.num_node_mlp = num_node_mlp

        self.use_edges = use_edges
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges
        self.num_edge_mlp = num_edge_mlp
        self.edge_dropout_rate = edge_dropout_rate

        self.aggregator = grouped_ops.grouped_scatter_add

        # node_model:
        # linear 1:
        # in_dim: 3*ndim + 2*edim (in_dim_edge) + gdim
        # hidden_dim: 4*ndim (in_dim)
        # linear 2: 
        # hidden_dim: 4*ndim (in_dim)
        # out_dim: ndim (out_dim)
        # linear 1, gelu act, layer_norm, linear 2

        self.node_model = MLP(
            in_dim=3*self.in_dim+2*self.in_dim_edges,
            hidden_dim=4*self.in_dim,
            out_dim=self.out_dim,
            layers=self.num_node_mlp,
            activation=self.activation_layer,
            normalization=self.normalization,
        )


        # edge_model:
        # linear 1:
        # in_dim: 2*ndim + edim (in_dim_edge) + gdim
        # hidden_dim: 4*edim (in_dim_edge)
        # linear 2: 
        # hidden_dim: 4*edim (in_dim_edge)
        # out_dim: edim (out_dim_edge)
        # linear 1, gelu act, layer_norm, linear 2, dropout

        self.edge_model = MLP(
            in_dim=2*self.in_dim+self.in_dim_edges,
            hidden_dim=4*self.in_dim_edges,
            out_dim=self.out_dim_edges,
            layers=self.num_edge_mlp,
            activation=self.activation_layer,
            last_dropout=self.edge_dropout_rate,
            normalization=self.normalization,
        )


    def gather_features(self, input_features, senders, receivers):
        out = []

        # we might need grouped_ops.grouped_gather(input_features, senders) on IPU
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


    def aggregate_features(self, input_features, senders, receivers, sender_features, receiver_features, dim):
        out = []
        aggregated_features = []

        if self.scatter_to in ['receivers', 'both']:
            # using direct_neighbour_aggregation to generate the message
            message = torch.cat([input_features, sender_features], dim=-1)
            # sum method is used with aggregators
            aggregated_features.append(self.aggregator(message, receivers, table_size=dim))

        if self.scatter_to in ['senders', 'both']:
            # using direct_neighbour_aggregation to generate the message
            message = torch.cat([input_features, receiver_features], dim=-1)
            # sum method is used with aggregators
            aggregated_features.append(self.aggregator(message, senders, table_size=dim))

        if self.node_combine_method == 'sum' and self.scatter_to == 'both':
            out.append(aggregated_features[0] + aggregated_features[1])
        elif self.scatter_to == 'both':
            out.append(torch.cat([aggregated_features[0] + aggregated_features[1]], dim=-1))
        else:
            out.extend(aggregated_features)

        return out


    def forward(self, batch):
        senders = batch.edge_index[0]
        receivers = batch.edge_index[1]
        node_org = batch.h
        edge_org = batch.edge_attr

        # ---------------EDGE step---------------
        edge_model_input, sender_nodes, receiver_nodes = self.gather_features(batch.h, senders, receivers)

        if self.use_edges:
            edge_model_input.append(batch.edge_attr)
            edge_model_input = torch.cat([edge_model_input[0], edge_model_input[1]], dim=-1)
            # edge dropout included in the edge_model
            batch.edge_attr = self.edge_model(edge_model_input)
        else:
            batch.edge_attr = edge_model_input

        # ---------------NODE step---------------
        # message + aggregate
        node_count_per_pack = batch.h.shape[-2]
        node_model_input = self.aggregate_features(batch.edge_attr, senders, receivers, sender_nodes, receiver_nodes,
                                                   node_count_per_pack)
        node_model_input.append(batch.h)
        batch.h = torch.cat([node_model_input[0], node_model_input[1]], dim=-1)
        batch.h = self.node_model(batch.h)

        # ---------------Apply norm activation and dropout---------------
        # use dropout value of the layer (default 0.35) and layer normalization
        batch.h = self.apply_norm_activation_dropout(batch.h, activation=False) + node_org
        batch.edge_attr = batch.edge_attr + edge_org

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
