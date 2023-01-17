from typing import Callable, Optional, Union

import torch
from goli.nn.base_graph_layer import BaseGraphModule
from goli.nn.base_layers import MLP
from goli.utils.decorators import classproperty


class MPNNPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        gather_from: str = "none",
        node_combine_method: str = "none",
        num_node_mlp: int = None,
        use_edges: bool = False,
        in_dim_edges: Optional[int] = None,
        out_dim_edges: Optional[int] = None,
        edge_dropout: Optional[Union[str, Callable]] = "none",
        num_edge_mlp: Optional[int] = None,
    ):
        r"""
            MPNNPyg: InteractionNetwork layer witg edges and global feature, GPS++ type of GNN layer
            GPS++: An Optimised Hybrid MPNN/Transformer for Molecular Property Prediction
            Dominic Masters, Josef Dean, Kerstin Klaser, Zhiyi Li, Sam Maddrell-Mander, Adam Sanders,
            Hatem Helal, Deniz Beker, Ladislav Rampášek, Dominique Beaini
            https://arxiv.org/abs/2212.02229

        # TODO: complete the doc string
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
            activation=activation,
            dropout=dropout,
            normalization=normalization,
        )

        self.gather_from = gather_from
        self.node_combine_method = node_combine_method
        self.num_node_mlp = num_node_mlp

        self.use_edges = use_edges
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges
        self.edge_dropout = edge_dropout
        self.num_edge_mlp = num_edge_mlp

        self.node_model = MLP(
            in_dim=self.in_dim,
            hidden_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=self.num_node_mlp,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )

        self.edge_model = MLP(
            in_dim=self.in_dim_edges,
            hidden_dim=self.in_dim,
            out_dim=self.out_dim_edges,
            layers=self.num_edge_mlp,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )


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
        senders = batch.edge_index[0]
        receivers = batch.edge_index[1]

        # ---------------EDGE step---------------
        edge_model_input, sender_nodes, receiver_nodes = self.gather_features(batch.h, senders, receivers)

        if self.use_edges:
            edge_model_input.append(batch.edge_attr)
            edge_model_input = torch.cat([edge_model_input[0], edge_model_input[1]], dim=-1)

            batch.edge_attr = self.edge_model(edge_model_input)
            # TODO: need to implement edge_dropout
            if self.edge_dropout:
                batch.edge_attr = self.edge_dropout(batch.edge_attr)
        else:
            batch.edge_attr = edge_model_input

        # ---------------NODE step---------------
        if self.use_edges:
            # TODO: implement self.edge_dna_dropout
            sender_nodes = self.edge_dna_dropout['senders'](sender_nodes)
            receiver_nodes = self.edge_dna_dropout['receivers'](receiver_nodes)
        # message + aggregate
        # TODO: implement self.aggregate_features
        node_model_input = self.aggregate_features(batch.edge_attr, senders, receivers, sender_nodes, receiver_nodes,
                                                   self.n_nodes_per_pack)
        node_model_input.append(batch.h)
        batch.h = torch.cat(node_model_input, axis=-1)
        batch.h = self.node_model(batch.h)

        # ---------------Apply norm activation and dropout---------------
        batch.h = self.apply_norm_activation_dropout(batch.h)
        # TODO: edge would only need a residual
        batch.edge_attr = self.apply_norm_activation_dropout(batch.edge_attr)

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
