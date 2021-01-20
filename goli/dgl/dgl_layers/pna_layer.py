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


class PNATower(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, graph_norm, batch_norm, aggregators, scalers, avg_d,
                 pretrans_layers, posttrans_layers, edge_features, edge_dim, activation):
        super().__init__()
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.edge_features = edge_features

        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.activation = activation

        self.aggregators = aggregators
        self.scalers = scalers
        self.pretrans = MLP(in_dim=2 * in_dim + (edge_dim if edge_features else 0), hidden_size=in_dim,
                            out_dim=in_dim, layers=pretrans_layers, mid_activation=activation, last_activation='none')
        self.posttrans = MLP(in_dim=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_dim=out_dim, layers=posttrans_layers, mid_activation=activation, last_activation='none')
        self.avg_d = avg_d

    def pretrans_edges(self, edges):
        if self.edge_features:
            z2 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['ef']], dim=1)
        else:
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        return {'e': self.pretrans(z2)}

    def message_func(self, edges):
        return {'e': edges.data['e']}

    def reduce_func(self, nodes):
        h = nodes.mailbox['e']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):
        g.ndata['h'] = h
        if self.edge_features: # add the edges information only if edge_features = True
            g.edata['ef'] = e

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization
        if self.graph_norm:
            h = h * snorm_n
        if self.batch_norm:
            h = self.batchnorm_h(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class PNALayer(BaseDGLLayer):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, graph_norm, batch_norm, activation,
                 towers=1, pretrans_layers=1, posttrans_layers=1, divide_input=True, residual=False, edge_features=False,
                 edge_dim=0):
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
        :param towers:              number of towers to use
        :param pretrans_layers:     number of layers in the transformation before the aggregation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        :param divide_input:        whether the input features should be split between towers or not
        :param residual:            whether to add a residual connection
        :param edge_features:       whether to use the edge features
        :param edge_dim:            size of the edge features
        """

                
        super().__init__(in_dim=in_dim, out_dim=out_dim, residual=residual, 
                activation=activation, dropout=dropout, batch_norm=batch_norm)
        
        assert ((not divide_input) or in_dim % towers == 0), "if divide_input is set the number of towers has to divide in_dim"
        assert (out_dim % towers == 0), "the number of towers has to divide the out_dim"

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in re.split('\s+|_', aggregators)]
        scalers = [SCALERS[scale] for scale in re.split('\s+|_', scalers)]

        self.divide_input = divide_input
        self.input_tower = in_dim // towers if divide_input else in_dim
        self.output_tower = out_dim // towers
        self.edge_features = edge_features
        self.graph_norm = graph_norm

        # convolution
        self.towers = nn.ModuleList()
        for _ in range(towers):
            self.towers.append(PNATower(in_dim=self.input_tower, out_dim=self.output_tower, aggregators=aggregators,
                                        scalers=scalers, avg_d=avg_d, pretrans_layers=pretrans_layers,
                                        posttrans_layers=posttrans_layers, batch_norm=batch_norm, dropout=dropout,
                                        graph_norm=graph_norm, edge_features=edge_features, edge_dim=edge_dim,
                                        activation=activation))
        # mixing network
        self.mixing_network = FCLayer(out_dim, out_dim, activation='LeakyReLU')

    def forward(self, g, h, e=None, snorm_n=None):
        h_in = h  # for residual connection

        if (e is None) and (self.edge_features):
            raise ValueError('edge features `e` required if `self.features` is True')

        if (snorm_n is None) and (self.graph_norm):
            raise ValueError('`snorm_n` required if `self.graph_norm` is True')

        if self.divide_input:
            h_cat = torch.cat(
                [tower(g, h[:, n_tower * self.input_tower: (n_tower + 1) * self.input_tower],
                       e, snorm_n)
                 for n_tower, tower in enumerate(self.towers)], dim=1)
        else:
            h_cat = torch.cat([tower(g, h, e, snorm_n) for tower in self.towers], dim=1)

        h_out = self.mixing_network(h_cat)

        if self.residual:
            h_out = h_in + h_out  # residual connection

        return h_out


    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)



class PNASimpleLayer(BaseDGLLayer):

    def __init__(self, in_dim, out_dim, aggregators, scalers, avg_d, dropout, batch_norm, activation,
                 posttrans_layers=1, residual=False):
        """
        A PNA layer that simply aggregates the neighbourhood (similar to GCN and GIN),
        without using the attention mechanism of the MPNN. It does not support edge features.

        :param in_dim:              size of the input per node
        :param out_dim:             size of the output per node
        :param aggregators:         set of aggregation function identifiers
        :param scalers:             set of scaling functions identifiers
        :param avg_d:               average degree of nodes in the training set, used by scalers to normalize
        :param dropout:             dropout used
        :param batch_norm:          whether to use batch normalisation
        :param posttrans_layers:    number of layers in the transformation after the aggregation
        """
        
        super().__init__(in_dim=in_dim, out_dim=out_dim, residual=residual, 
                activation=activation, dropout=dropout, batch_norm=batch_norm)

        # retrieve the aggregators and scalers functions
        aggregators = [AGGREGATORS[aggr] for aggr in re.split('\s+|_', aggregators)]
        scalers = [SCALERS[scale] for scale in re.split('\s+|_', scalers)]

        self.aggregators = aggregators
        self.scalers = scalers

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.posttrans = MLP(in_dim=(len(aggregators) * len(scalers) + 1) * in_dim, hidden_size=out_dim,
                             out_dim=out_dim, layers=posttrans_layers, 
                             mid_activation=activation, last_activation=activation,
                             dropout=dropout, mid_batch_norm=batch_norm, last_batch_norm=batch_norm)
        self.avg_d = avg_d


    def reduce_func(self, nodes):
        h = nodes.mailbox['m']
        D = h.shape[-2]
        h = torch.cat([aggregate(h) for aggregate in self.aggregators], dim=1)
        h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=1)
        return {'h': h}


    def forward(self, g, h):
        g.ndata['h'] = h

        # aggregation
        g.update_all(fn.copy_u('h', 'm'), self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        return h


    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_dim, self.out_dim)



