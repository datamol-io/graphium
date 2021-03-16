import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Tuple, Union, Callable
from dgl import DGLGraph

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer
from goli.utils.decorators import classproperty

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class GatedGCNLayer(BaseDGLLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        out_dim_edges: int,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        r"""
        ResGatedGCN: Residual Gated Graph ConvNets
        An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
        https://arxiv.org/pdf/1711.07553v2.pdf

        Parameters:

            in_dim:
                Input feature dimensions of the layer

            out_dim:
                Output feature dimensions of the layer, and for the edges

            in_dim_edges:
                Input edge-feature dimensions of the layer

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

        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim_edges, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)

    def message_func(self, edges):
        Bh_j = edges.src["Bh"]
        e_ij = edges.data["Ce"] + edges.src["Dh"] + edges.dst["Eh"]  # e_ij = Ce_ij + Dhi + Ehj
        edges.data["e"] = e_ij
        return {"Bh_j": Bh_j, "e_ij": e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data["Ah"]
        Bh_j = nodes.mailbox["Bh_j"]
        e = nodes.mailbox["e_ij"]
        sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)
        # h = Ah_i + torch.mean( sigma_ij * Bh_j, dim=1 ) # hi = Ahi + mean_j alpha_ij * Bhj
        h = Ah_i + torch.sum(sigma_ij * Bh_j, dim=1) / (
            torch.sum(sigma_ij, dim=1) + 1e-6
        )  # hi = Ahi + sum_j eta_ij/sum_j' eta_ij' * Bhj <= dense attention
        return {"h": h}

    def forward(self, g: DGLGraph, h: torch.Tensor, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply the graph convolutional layer, with the specified activations,
        normalizations and dropout.

        Parameters:

            g:
                graph on which the convolution is done

            h: `torch.Tensor[..., N, Din]`
                Node feature tensor, before convolution.
                N is the number of nodes, Din is the input dimension ``self.in_dim``

            e: `torch.Tensor[..., N, Din_edges]`
                Edge feature tensor, before convolution.
                N is the number of nodes, Din is the input edge dimension  ``self.in_dim_edges``

        Returns:
            `torch.Tensor[..., N, Dout]`:
                Node feature tensor, after convolution.
                N is the number of nodes, Dout is the output dimension ``self.out_dim``

            `torch.Tensor[..., N, Dout]`:
                Edge feature tensor, after convolution.
                N is the number of nodes, Dout_edges is the output edge dimension ``self.out_dim``

        """

        g.ndata["h"] = h
        g.ndata["Ah"] = self.A(h)
        g.ndata["Bh"] = self.B(h)
        g.ndata["Dh"] = self.D(h)
        g.ndata["Eh"] = self.E(h)
        g.edata["e"] = e
        g.edata["Ce"] = self.C(e)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata["h"]  # result of graph convolution
        e = g.edata["e"]  # result of graph convolution

        h = self.apply_norm_activation_dropout(h)
        e = self.apply_norm_activation_dropout(e)

        return h, e

    @classproperty
    def layer_supports_edges(cls) -> bool:
        r"""
        Return a boolean specifying if the layer type supports edges or not.

        Returns:

            bool:
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
