"""
Unit tests for the positional encodings in goli/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

from goli.features.positional_encoding import node_to_edge, node_to_pair, pair_to_node, pair_to_edge


class test_positional_encodings(ut.TestCase):

    # 4-barbell
    H = nx.barbell_graph(4, 0)
    adj = nx.to_numpy_array(H)
    num_nodes, num_feat = 8, 5
    node_pe = np.random.rand(num_nodes, num_feat)


    def test_dimensions(self):
        edge_pe1 = node_to_edge(self.node_pe, self.adj)
        pair_pe = node_to_pair(self.node_pe, self.num_nodes)
        feat_list = []
        for dim in range(pair_pe.shape[-1]):
            feat_list.append(pair_to_edge(pair_pe[..., dim], self.adj))
        edge_pe2 = np.stack(feat_list, axis=-1)
        np.testing.assert_array_almost_equal(edge_pe1, edge_pe2)


if __name__ == "__main__":
    ut.main()