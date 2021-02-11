import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union, Callable

import dgl.function as fn
from dgl.nn.pytorch import GraphConv
from dgl import DGLGraph

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer
from goli.commons.decorators import classproperty

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""


class GCNLayer(BaseDGLLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[str, Callable] = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        r"""
        Graph convolutional network (GCN) layer from
        Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
        http://arxiv.org/abs/1609.02907

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer

            activation:
                activation function to use in the layer

            dropout:
                The ratio of units to dropout. Must be between 0 and 1

            batch_norm:
                Whether to use batch normalization
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.conv = GraphConv(
            in_feats=in_dim,
            out_feats=out_dim,
            norm="both",
            weight=True,
            bias=True,
            activation=None,
            allow_zero_in_degree=False,
        )

    def forward(self, g: DGLGraph, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the graph convolutional layer, with the specified activations,
        normalizations and dropout.

        Parameters:

            g:
                graph on which the convolution is done

            h: `torch.Tensor[..., N, Din]`
                Node feature tensor, before convolution.
                N is the number of nodes, Din is the input dimension ``self.in_dim``

        Returns:

            `torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution.
                N is the number of nodes, Dout is the output dimension ``self.out_dim``

        """

        h = self.conv(g, h)
        h = self.apply_norm_activation_dropout(h)

        return h

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            bool
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
