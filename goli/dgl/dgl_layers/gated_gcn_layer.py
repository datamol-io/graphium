import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer

"""
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and Thomas Laurent, ICLR 2018)
    https://arxiv.org/pdf/1711.07553v2.pdf
"""


class GatedGCNLayer(BaseDGLLayer):
    """
    Param: []
    """

    def __init__(self, in_dim, out_dim, in_dim_e, out_dim_e, dropout, batch_norm, activation, residual=False):

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            residual=residual,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.in_channels = in_dim
        self.out_channels = out_dim

        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim_e, out_dim_e, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)
        self.bn_node_h = nn.BatchNorm1d(out_dim)
        self.bn_node_e = nn.BatchNorm1d(out_dim)

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

    def forward(self, g, h, e):

        h_in = h  # for residual connection
        e_in = e  # for residual connection

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

        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization
            e = self.bn_node_e(e)  # batch normalization

        h = self.activation(h)  # non-linear activation
        e = self.activation(e)  # non-linear activation

        if self.residual:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection

        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)

        return h, e

    def __repr__(self):
        return "{}(in_channels={}, out_channels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )

    @staticmethod
    def _parse_layer_args(in_dims, out_dims, **kwargs):

        kwargs_of_lists = {}
        kwargs_keys_to_remove = ["in_dim_edges"]
        in_dim_e = deepcopy(in_dims)
        in_dim_e[0] = kwargs["in_dim_edges"]

        kwargs_of_lists["in_dim_e"] = in_dim_e
        kwargs_of_lists["out_dim_e"] = out_dims
        true_out_dims = in_dims[1:] + out_dims[-1:]

        return in_dims, out_dims, true_out_dims, kwargs_of_lists, kwargs_keys_to_remove
