from typing import Callable, Optional, Union, Tuple, List

import torch
from torch import Tensor, IntTensor, LongTensor
from graphium.nn.base_graph_layer import BaseGraphModule
from graphium.nn.base_layers import MLP
from graphium.utils.decorators import classproperty
from torch_geometric.nn.aggr import MultiAggregation, Aggregation
from torch_geometric.data import Batch


class MPNNPlusPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int = 64,
        out_dim: int = 64,
        activation: Union[str, Callable] = "gelu",
        dropout: float = 0.3,
        normalization: Union[str, Callable] = "layer_norm",
        gather_from: str = "both",
        scatter_to: str = "both",
        node_combine_method: str = "concat",
        num_node_mlp: int = 2,
        mlp_expansion_ratio: int = 2,
        use_edges: bool = True,
        in_dim_edges: Optional[int] = 32,
        out_dim_edges: Optional[int] = 32,
        aggregation_method: Optional[List[Union[str, Aggregation]]] = ["sum"],
        num_edge_mlp: Optional[int] = 2,
        use_globals: bool = True,
        edge_dropout_rate: Optional[float] = 0.0035,
        **kwargs,
    ):
        r"""
            MPNNPlusPyg: InteractionNetwork layer witg edges and global feature, GPS++ type of GNN layer
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

            aggregation_method:
                Methods for aggregating (scatter) messages built from node and edge features.
                Provide a list of `Aggregation` or strings.
                supported strings are:

                - "sum" / "add" (Default)
                - "mean"
                - "max"
                - "min"
                - "softmax"
                - "median"
                - "std"
                - "var"

            num_node_mlp:
                Number of mlp layer used for node model

            mlp_expansion_ratio:
                Expansion ratio for node and edge mlp

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
            **kwargs,
        )

        self.gather_from = gather_from
        self.scatter_to = scatter_to
        self.node_combine_method = node_combine_method
        self.num_node_mlp = num_node_mlp
        self.mlp_expansion_ratio = mlp_expansion_ratio

        self.use_edges = use_edges
        self.in_dim_edges = in_dim_edges
        self.out_dim_edges = out_dim_edges
        self.num_edge_mlp = num_edge_mlp
        self.edge_dropout_rate = edge_dropout_rate

        self.aggregator = MultiAggregation(aggregation_method)

        # node_model:
        edge_dim = self.out_dim_edges if use_edges else self.in_dim_edges
        if self.node_combine_method == "concat":
            node_model_in_dim = 3 * self.in_dim + 2 * edge_dim
        elif self.node_combine_method == "sum":
            node_model_in_dim = 2 * self.in_dim + edge_dim
        else:
            raise ValueError(f"node_combine_method {self.node_combine_method} not recognised.")
        node_model_hidden_dim = self.mlp_expansion_ratio * self.in_dim
        self.node_model = MLP(
            in_dim=node_model_in_dim,
            hidden_dims=node_model_hidden_dim,
            out_dim=self.out_dim,
            depth=self.num_node_mlp,
            activation=self.activation_layer,
            last_dropout=self.dropout,
            normalization=self.normalization,
        )

        # edge_model:
        if self.node_combine_method == "concat":
            edge_model_in_dim = 2 * self.in_dim + self.in_dim_edges
        elif self.node_combine_method == "sum":
            edge_model_in_dim = self.in_dim + self.in_dim_edges
        else:
            raise ValueError(f"node_combine_method {self.node_combine_method} not recognised.")
        edge_model_hidden_dim = self.mlp_expansion_ratio * self.in_dim_edges
        self.edge_model = MLP(
            in_dim=edge_model_in_dim,
            hidden_dims=edge_model_hidden_dim,
            out_dim=self.out_dim_edges,
            depth=self.num_edge_mlp,
            activation=self.activation_layer,
            last_dropout=self.edge_dropout_rate,
            normalization=self.normalization,
        )

        self.use_globals = use_globals

    def gather_features(
        self,
        input_features: Tensor,
        senders: Union[IntTensor, LongTensor],
        receivers: Union[IntTensor, LongTensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Function to gather node features based on the senders and receivers of the edge indices.

        Parameters:

            input_features:
                Node features of the batch

            senders:
                Senders of the edge_index of the batch

            receivers:
                Receivers of the edge_index of the batch

        Output:
            Gathered node features (sender and receiver) summed up or concatenated
            Gathered sender features
            Gathered receiver features
        """

        out = []

        receiver_features = input_features[receivers]
        sender_features = input_features[senders]

        if self.gather_from == "receivers":
            out.append(receiver_features)

        if self.gather_from == "senders":
            out.append(sender_features)

        if self.gather_from == "both":
            if self.node_combine_method == "sum":
                out.append(receiver_features + sender_features)
            elif self.node_combine_method == "concat":
                out.append(torch.cat([receiver_features, sender_features], dim=-1))
            else:
                raise ValueError(f"node_combine_method {self.node_combine_method} not recognised.")

        return out, sender_features, receiver_features

    def aggregate_features(
        self,
        input_features: Tensor,
        senders: Union[IntTensor, LongTensor],
        receivers: Union[IntTensor, LongTensor],
        sender_features: Tensor,
        receiver_features: Tensor,
        size: int,
    ) -> Tensor:
        r"""
        Function to aggregate (scatter) messages built from node and edge features.

        Parameters:

            input_features:
                Edge features of the batch

            senders:
                Senders of the edge_index of the batch

            receivers:
                Receivers of the edge_index of the batch

            sender_features:
                Senders features gathered from the gather_features function

            receiver_features:
                Receiver features gathered from the gather_features function

            size:
                size of the aggregation, equals to the total number of nodes

        Returns:
            Tensor:
                Aggregated node features

        """

        out = []
        aggregated_features = []

        if self.scatter_to in ["receivers", "both"]:
            # using direct_neighbour_aggregation to generate the message
            message = torch.cat([input_features, sender_features], dim=-1)
            # sum method is used with aggregators
            aggregated_features.append(self.aggregator(message, receivers, dim_size=size))

        if self.scatter_to in ["senders", "both"]:
            # using direct_neighbour_aggregation to generate the message
            message = torch.cat([input_features, receiver_features], dim=-1)
            # sum method is used with aggregators
            aggregated_features.append(self.aggregator(message, senders, dim_size=size))

        if self.node_combine_method == "sum" and self.scatter_to == "both":
            out.append(aggregated_features[0] + aggregated_features[1])
        elif self.scatter_to == "both":
            out.append(torch.cat([aggregated_features[0], aggregated_features[1]], dim=-1))
        else:
            out.extend(aggregated_features)

        return out

    def forward(self, batch: Batch) -> Batch:
        r"""
        Forward function of the MPNN Plus layer
        Parameters:
            batch:
                pyg Batch graph to pass through the layer
        Returns:
            batch:
                pyg Batch graph with updated node and edge features
        """
        senders = batch.edge_index[0]
        receivers = batch.edge_index[1]
        # ---------------EDGE step---------------
        edge_model_input, sender_nodes, receiver_nodes = self.gather_features(batch.feat, senders, receivers)

        if self.use_edges:
            edge_model_input.append(batch.edge_feat)
            edge_model_input = torch.cat([edge_model_input[0], edge_model_input[1]], dim=-1)
            # edge dropout included in the edge_model
            batch.edge_feat = self.edge_model(edge_model_input)
        else:
            batch.edge_feat = edge_model_input

        # ---------------NODE step---------------
        # message + aggregate
        node_count_per_pack = batch.feat.shape[-2]
        node_model_input = self.aggregate_features(
            batch.edge_feat, senders, receivers, sender_nodes, receiver_nodes, node_count_per_pack
        )
        node_model_input.append(batch.feat)
        batch.feat = torch.cat([node_model_input[0], node_model_input[1]], dim=-1)
        batch.feat = self.node_model(batch.feat)

        # ---------------Apply norm activation and dropout---------------
        # use dropout value of the layer (default 0.3)
        batch.feat = self.apply_norm_activation_dropout(
            batch.feat,
            normalization=False,
            activation=False,
            batch_idx=batch.batch,
            batch_size=batch.num_graphs,
        )

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
                Always ``True`` for the current class
        """
        return True

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
