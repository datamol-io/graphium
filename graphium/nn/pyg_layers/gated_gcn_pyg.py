"""
Unit tests for the different layers of graphium/nn/pyg_layers/...

The layers are not thoroughly tested due to the difficulty of testing them

adapated from https://github.com/rampasek/GraphGPS/blob/main/graphgps/layer/gps_layer.py
"""

from typing import Union, Callable
from functools import partial

import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter
from torch_geometric.data import Data, Batch

from graphium.nn.base_graph_layer import BaseGraphStructure, check_intpus_allow_int
from graphium.nn.base_layers import FCLayer
from graphium.utils.decorators import classproperty


class GatedGCNPyg(MessagePassing, BaseGraphStructure):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        out_dim_edges: int = None,
        activation: Union[Callable, str] = "relu",
        dropout: float = 0.0,
        normalization: Union[str, Callable] = "none",
        eps: float = 1e-5,
        **kwargs,
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

            normalization:
                Normalization to use. Choices:

                - "none" or `None`: No normalization
                - "batch_norm": Batch normalization
                - "layer_norm": Layer normalization
                - `Callable`: Any callable function

            eps:
                Epsilon value for the normalization `sum(gate_weights * messages) / (sum(gate_weights) + eps)`,
                where `gate_weights` are the weights of the gates and follow a sigmoid function.

        """
        MessagePassing.__init__(self, aggr="add", flow="source_to_target", node_dim=-2)
        BaseGraphStructure.__init__(
            self,
            in_dim=in_dim,
            out_dim=out_dim,
            activation=activation,
            dropout=dropout,
            normalization=normalization,
            **kwargs,
        )

        self._initialize_activation_dropout_norm()

        # Allow int32 in the edge_index
        self.__check_input__ = partial(check_intpus_allow_int, self)
        if out_dim_edges is None:
            out_dim_edges = in_dim_edges

        # Initialize the layers for the gating
        self.A = FCLayer(in_dim, out_dim, activation=None, bias=True)
        self.B = FCLayer(in_dim, out_dim, activation=None, bias=True)
        self.C = FCLayer(in_dim_edges, out_dim, activation=None, bias=True)
        self.D = FCLayer(in_dim, out_dim, activation=None, bias=True)
        self.E = FCLayer(in_dim, out_dim, activation=None, bias=True)

        self.edge_out = FCLayer(
            in_dim=out_dim, out_dim=out_dim_edges, activation=None, dropout=dropout, bias=True
        )
        self.eps = eps

    def forward(
        self,
        batch: Union[Data, Batch],
    ) -> Union[Data, Batch]:
        r"""
        Forward pass the Gated GCN layer
        extract the following from the batch:
        x, node features with dim [n_nodes, in_dim]
        e, edge features with dim [n_edges, in_dim]
        edge_index with dim [2, n_edges]

        Parameters:
            batch: pyg Batch graph to pass through the layer
        Returns:
            batch: pyg Batch graph
        """

        x, e, edge_index = batch.feat, batch.edge_feat, batch.edge_index

        # Apply the linear layers
        Ax = self.A(x)
        Bx = self.B(x)
        Ce = self.C(e)
        Dx = self.D(x)
        Ex = self.E(x)

        # Propagate, and apply norm, activation, dropout
        x, e = self.propagate(edge_index, Bx=Bx, Dx=Dx, Ex=Ex, Ce=Ce, e=e, Ax=Ax)
        x = self.apply_norm_activation_dropout(x, batch_idx=batch.batch)
        e = self.edge_out(e)

        # Output
        batch.feat = x
        batch.edge_feat = e

        return batch

    def message(self, Dx_i: torch.Tensor, Ex_j: torch.Tensor, Ce: torch.Tensor) -> torch.Tensor:
        """
        message function
        Parameters:
            Dx_i: tensor with dimension [n_edges, out_dim]
            Ex_j: tensor with dimension [n_edges, out_dim]
            Ce: tensor with dimension [n_edges, out_dim]
        Returns:
            sigma_ij: tensor with dimension [n_edges, out_dim]
        """
        e_ij = Dx_i + Ex_j + Ce
        sigma_ij = torch.sigmoid(e_ij)

        self.e = e_ij
        return sigma_ij

    def aggregate(
        self,
        sigma_ij: torch.Tensor,
        index: torch.Tensor,
        Bx_j: torch.Tensor,
        Bx: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        aggregation function of the layer
        Parameters:
            sigma_ij: the output from message() function with dim [n_edges, out_dim]
            index: dim [n_edges]
            Bx_j: dim [n_edges, out_dim]
            Bx: dim [n_nodes, out_dim]
        Returns:
            out: dim [n_nodes, out_dim]
        """
        dim_size = Bx.shape[0]

        # Sum the messages, weighted by the gates. Sum the gates.
        numerator_eta_xj = scatter(sigma_ij * Bx_j, index, 0, None, dim_size, reduce="sum")
        denominator_eta_xj = scatter(sigma_ij, index, 0, None, dim_size, reduce="sum")

        # Cast to float32 if needed
        dtype = denominator_eta_xj.dtype
        if dtype == torch.float16:
            numerator_eta_xj = numerator_eta_xj.to(dtype=torch.float32)
            denominator_eta_xj = denominator_eta_xj.to(dtype=torch.float32)

        # Normalize the messages by the sum of the gates
        out = numerator_eta_xj / (denominator_eta_xj + self.eps)

        # Cast back to float16 if needed
        if dtype == torch.float16:
            out = out.to(dtype=dtype)
        return out

    def update(self, aggr_out: torch.Tensor, Ax: torch.Tensor):
        r"""
        update function of the layer
        Parameters:
            aggr_out: the output from aggregate() function with dim [n_nodes, out_dim]
            Ax: tensor with dim [n_nodes, out_dim]
        Returns:
            x: dim [n_nodes, out_dim]
            e_out: dim [n_edges, out_dim_edges]
        """
        x = Ax + aggr_out
        e_out = self.e
        del self.e
        return x, e_out

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
