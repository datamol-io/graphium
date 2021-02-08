import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from dgl import DGLGraph

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer
from goli.commons.decorators import classproperty

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class GATLayer(BaseDGLLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads,
        activation="elu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        """
        GAT: Graph Attention Network
        Graph Attention Networks (Veličković et al., ICLR 2018)
        https://arxiv.org/abs/1710.10903

        The implementation is built on top of the DGL ``GATCONV`` layer

        Parameters:

            in_dim: int
                Input feature dimensions of the layer

            out_dim: int
                Output feature dimensions of the layer

            num_heads: int
                Number of heads in Multi-Head Attention

            activation: str, Callable
                activation function to use in the layer

            dropout: float
                The ratio of units to dropout. Must be between 0 and 1

            batch_norm: bool
                Whether to use batch normalization
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.num_heads = num_heads

        self.gatconv = GATConv(
            in_feats=self.in_dim,
            out_feats=self.out_dim,
            num_heads=self.num_heads,
            feat_drop=self.dropout,
            attn_drop=self.dropout,
            activation=None,  # Activation is applied after
        )

    def forward(self, g: DGLGraph, h: torch.Tensor) -> torch.Tensor:
        """
        Apply the graph convolutional layer, with the specified activations,
        normalizations and dropout.

        Parameters:

            g: dgl.DGLGraph
                graph on which the convolution is done

            h: torch.Tensor(..., N, Din)
                Node feature tensor, before convolution.
                N is the number of nodes, Din is the input dimension ``self.in_dim``

            Returns:

            h: torch.Tensor(..., N, Dout)
                Node feature tensor, after convolution.
                N is the number of nodes, Dout is the output dimension ``self.out_dim``

        """

        h = self.gatconv(g, h).flatten(1)
        self.apply_norm_activation_dropout(h, batch_norm=True, activation=True, dropout=False)

        return h

    @classproperty
    def layer_supports_edges(cls) -> bool:
        """
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            supports_edges: bool
                Always ``False`` for the current class
        """
        return False

    @property
    def layer_inputs_edges(self) -> bool:
        """
        Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            uses_edges: bool
                Always ``False`` for the current class
        """
        return False

    @property
    def layer_outputs_edges(self) -> bool:
        """
        Abstract method. Return a boolean specifying if the layer type
        uses edges as input or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns:

            uses_edges: bool
                Always ``False`` for the current class
        """
        return False

    @property
    def out_dim_factor(self) -> int:
        """
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns:

            dim_factor: int
                Always ``self.num_heads`` for the current class
        """
        return self.num_heads
