from typing import Callable, Union
from functools import partial

import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, Batch

from graphium.nn.base_graph_layer import BaseGraphModule, check_intpus_allow_int
from graphium.utils.decorators import classproperty


class GCNConvPyg(BaseGraphModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        **kwargs,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            **kwargs,
        )

        self.model = pyg_nn.GCNConv(
            in_channels=self.in_dim, out_channels=out_dim, add_self_loops=False, normalize=False
        )
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
