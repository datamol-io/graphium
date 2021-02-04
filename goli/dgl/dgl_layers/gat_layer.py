import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer

"""
    GAT: Graph Attention Network
    Graph Attention Networks (Veličković et al., ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""


class GATLayer(BaseDGLLayer):
    """
    Parameters
    ----------
    in_dim :
        Number of input features.
    out_dim :
        Number of output features.
    num_heads : int
        Number of heads in Multi-Head Attention.
    dropout :
        Required for dropout of attn and feat in GATConv
    batch_norm :
        boolean flag for batch_norm layer.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.

    Using dgl builtin GATConv by default:
    https://github.com/graphdeeplearning/benchmarking-gnns/commit/206e888ecc0f8d941c54e061d5dffcc7ae2142fc
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads,
        activation="elu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):

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

    def forward(self, g, h):

        h = self.gatconv(g, h).flatten(1)
        self.apply_norm_activation_dropout(h, batch_norm=True, activation=True, dropout=False)

        return h

    @staticmethod
    def layer_supports_edges() -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns
        ---------

        supports_edges: bool
            Always ``False`` for the current class
        """
        return False

    @abc.abstractmethod
    def layer_uses_edges(self) -> bool:
        r"""
        Return a boolean specifying if the layer type
        uses edges or not.
        It is different from ``layer_supports_edges`` since a layer that
        supports edges can decide to not use them.

        Returns
        ---------

        uses_edges: bool
            Always ``False`` for the current class
        """
        return False

    @abc.abstractmethod
    def get_out_dim_factor(self) -> int:
        r"""
        Get the factor by which the output dimension is multiplied for
        the next layer.

        For standard layers, this will return ``1``.

        But for others, such as ``GatLayer``, the output is the concatenation
        of the outputs from each head, so the out_dim gets multiplied by
        the number of heads, and this function should return the number
        of heads.

        Returns
        ---------

        dim_factor: int
            Always ``self.num_heads`` for the current class
        """
        return self.num_heads
