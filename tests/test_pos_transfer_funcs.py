"""
Unit tests for the positional encodings in graphium/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

from graphium.features.spectral import compute_laplacian_pe
from graphium.features.transfer_pos_level import (
    node_to_edge,
    node_to_nodepair,
    edge_to_nodepair,
    nodepair_to_node,
    nodepair_to_edge,
    graph_to_node,
)


class test_pos_transfer_funcs(ut.TestCase):
    # 4-barbell
    G = nx.barbell_graph(4, 0)
    adj = nx.to_numpy_array(G)
    num_nodes, num_feat = 8, 5
    node_pe = np.random.rand(num_nodes, num_feat)

    def test_different_pathways_from_node_to_edge(self):
        edge_pe1, _ = node_to_edge(self.node_pe, self.adj, {})
        nodepair_pe1 = node_to_nodepair(self.node_pe, self.num_nodes)
        edge_pe2, _ = nodepair_to_edge(nodepair_pe1, self.adj, {})
        nodepair_pe2, _ = edge_to_nodepair(edge_pe1, self.adj, self.num_nodes, {})
        edge_pe3, _ = nodepair_to_edge(nodepair_pe2, self.adj, {})
        np.testing.assert_array_almost_equal(edge_pe1, edge_pe2)
        np.testing.assert_array_almost_equal(edge_pe1, edge_pe3)


if __name__ == "__main__":
    ut.main()
