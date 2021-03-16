import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl import DGLGraph
from typing import Callable, Union

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer
from goli.dgl.base_layers import MLP
from goli.utils.decorators import classproperty

"""
    GIN: Graph Isomorphism Networks
    HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
    https://arxiv.org/pdf/1810.00826.pdf
"""


class GINLayer(BaseDGLLayer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        init_eps: float = 0.0,
        learn_eps: bool = True,
    ):
        r"""
        GIN: Graph Isomorphism Networks
        HOW POWERFUL ARE GRAPH NEURAL NETWORKS? (Keyulu Xu, Weihua Hu, Jure Leskovec and Stefanie Jegelka, ICLR 2019)
        https://arxiv.org/pdf/1810.00826.pdf

        [!] code adapted from dgl implementation of GINConv

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
            batch_norm=batch_norm,
        )

        # Specify to consider the edges weight in the aggregation

        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer("eps", torch.FloatTensor([init_eps]))

        # The weights of the model, applied after the aggregation
        self.mlp = MLP(
            in_dim=self.in_dim,
            hidden_dim=self.in_dim,
            out_dim=self.out_dim,
            layers=2,
            actibation=self.activation_layer,
            last_activation="none",
            batch_norm=self.batch_norm,
            last_batch_norm=False,
        )

    def message_func(self, g):
        r"""
        If edge weights are provided, use them to weight the messages
        """

        if "w" in g.edata.keys():
            func = fn.u_mul_e("h", "w", "m")
        else:
            func = fn.copy_u("h", "m")
        return func

    def forward(self, g: DGLGraph, h: torch.Tensor) -> torch.Tensor:
        r"""
        Apply the GIN convolutional layer, with the specified activations,
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

        # Aggregate the message
        g = g.local_var()
        g.ndata["h"] = h
        func = fn.copy_u("h", "m")
        g.update_all(self.message_func(g), fn.sum("m", "neigh"))
        h = (1 + self.eps) * h + g.ndata["neigh"]

        # Apply the MLP
        h = self.mlp(h)
        h = self.apply_norm_activation_dropout(h)

        return h

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
