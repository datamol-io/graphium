from typing import Callable, Union, Optional
from functools import partial

import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

from graphium.nn.base_graph_layer import BaseGraphModule, check_intpus_allow_int
from graphium.nn.base_layers import MLP
from graphium.utils.decorators import classproperty


class GINConvPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        **kwargs,
    ):
        r"""
        GIN: Graph Isomorphism Networks
        HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
        https://arxiv.org/pdf/1810.00826.pdf

        [!] code uses the pytorch-geometric implementation of GINConv

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
            **kwargs,
        )

        gin_nn = MLP(
            in_dim=self.in_dim,
            hidden_dims=self.in_dim,
            out_dim=self.out_dim,
            depth=2,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )

        self.model = pyg_nn.GINConv(gin_nn)
        self.model.__check_input__ = partial(check_intpus_allow_int, self)

    def forward(
        self,
        batch: Union[Data, Batch],
    ) -> Union[Data, Batch]:
        r"""
        forward function of the layer
        Parameters:
            batch: pyg Batch graphs to pass through the layer
        Returns:
            batch: pyg Batch graphs
        """
        batch.feat = self.model(batch.feat, batch.edge_index)
        batch.feat = self.apply_norm_activation_dropout(batch.feat, batch_idx=batch.batch)

        return batch

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            supports_edges: bool
                Always ``False`` for the current class
        """
        return False

    @property
    def layer_inputs_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            bool:
                Always ``False`` for the current class
        """
        return False

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


class GINEConvPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: Optional[int] = None,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        **kwargs,
    ):
        r"""
        GINE: Graph Isomorphism Networks with Edges
        Strategies for Pre-training Graph Neural Networks
        Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, Jure Leskovec
        https://arxiv.org/abs/1905.12265

        [!] code uses the pytorch-geometric implementation of GINEConv

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

        gin_nn = MLP(
            in_dim=self.in_dim,
            hidden_dims=self.in_dim,
            out_dim=self.out_dim,
            depth=2,
            activation=self.activation_layer,
            last_activation="none",
            normalization=self.normalization,
            last_normalization="none",
        )

        self.model = pyg_nn.GINEConv(gin_nn, edge_dim=in_dim_edges)  # , node_dim=-1)
        self.model.__check_input__ = partial(check_intpus_allow_int, self)

    def forward(
        self,
        batch: Union[Data, Batch],
    ) -> Union[Data, Batch]:
        r"""
        forward function of the layer
        Parameters:
            batch: pyg Batch graphs to pass through the layer
        Returns:
            batch: pyg Batch graphs
        """
        batch.feat = self.model(batch.feat, batch.edge_index, batch.edge_feat)
        batch.feat = self.apply_norm_activation_dropout(batch.feat, batch_idx=batch.batch)

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
