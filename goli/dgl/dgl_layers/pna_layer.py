import re
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn

from goli.dgl.pna_operations import AGGREGATORS, SCALERS
from goli.dgl.base_layers import MLP, FCLayer
from goli.dgl.dgl_layers.base_dgl_layer import BaseDGLLayer

"""
    PNA: Principal Neighbourhood Aggregation 
    Gabriele Corso, Luca Cavalleri, Dominique Beaini, Pietro Lio, Petar Velickovic
    https://arxiv.org/abs/2004.05718
"""


class BasePNALayer(BaseDGLLayer):
    def __init__(
        self,
        in_dim,
        out_dim,
        aggregators,
        scalers,
        avg_d,
        residual,
        activation,
        dropout,
        batch_norm,
        graph_norm,
    ):
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            residual=residual,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        # Initializing basic attributes
        self.graph_norm = graph_norm
        self.edge_features = edge_features
        self.avg_d = avg_d

        # Initializing aggregators and scalers
        self.aggregators = [AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [SCALERS[scale] for scale in scalers]

        # Initilizing batch_norm layer
        if self.batch_norm:
            self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def message_func(self, edges):
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes):
        h_in = nodes.data["h"]
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        to_cat = []
        for aggregate in self.aggregators:
            try:
                to_cat.append(aggregate(self, h))
            except:
                to_cat.append(aggregate(self, h, h_in))

        h = torch.cat(to_cat, dim=1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)

        return {"h": h}

    def post_forward(self, h, h_in, snorm_n):
        # graph and batch normalization, residual and dropout
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)

        return h


class PNAComplexLayer(BasePNALayer):
    def __init__(
        self,
        in_dim,
        out_dim,
        aggregators,
        scalers,
        avg_d,
        dropout,
        graph_norm,
        batch_norm,
        activation,
        residual=False,
        pretrans_layers=1,
        posttrans_layers=1,
        edge_dim=0,
    ):

        """
        A PNA layer that simply aggregates the neighbourhood using an MPNN-inspired
        architecture for the attention mechanism and edge features.

        :param in_dim:              size of the input per node
        :param out_dim:             size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param dropout:             dropout used
        :param graph_norm:          whether to use graph normalisation
        :param batch_norm:          whether to use batch normalisation
        :param pretrans_layers:     number of layers in the transformation before the aggregation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        :param edge_dim:            size of the edge features
        """

        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            residual=residual,
            aggregators=aggregators,
            scalers=scalers,
            avg_d=avg_d,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            graph_norm=graph_norm,
        )

        self.edge_dim = edge_dim
        self.edge_features = self.edge_dim > 0
        self.pretrans = MLP(
            in_dim=2 * in_dim + edge_dim,
            hidden_dim=in_dim,
            out_dim=in_dim,
            layers=pretrans_layers,
            mid_activation=self.activation,
            last_activation="none",
        )
        self.posttrans = MLP(
            in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim,
            hidden_dim=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=self.activation,
            last_activation="none",
        )

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src["h"], edges.dst["h"], edges.data["ef"]], dim=1)
        else:
            z2 = torch.cat([edges.src["h"], edges.dst["h"]], dim=1)
        return {"e": self.pretrans(z2)}

    def forward(self, g, h, e, snorm_n):

        h_in = h
        g.ndata["h"] = h

        if self.edge_features:  # add the edges information only if edge_features = True
            g.edata["ef"] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata["h"]], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual
        h = self.post_forward(h=h, h_in=h_in, snorm_n=snorm_n)

        return h
