"""
Unit tests for the positional encodings in goli/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

from goli.features.spectral import compute_laplacian_pe
from goli.features.transfer_pos_level import (
    node_to_edge,
    node_to_nodepair,
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
        nodepair_pe = node_to_nodepair(self.node_pe, self.num_nodes)
        edge_pe2 = nodepair_to_edge(nodepair_pe, self.adj, {})
        np.testing.assert_array_almost_equal(edge_pe1, edge_pe2)

    def test_graph_to_node(self):
        num_pos = 2
        graph_pe, _, _ = compute_laplacian_pe(self.adj, num_pos, cache={}, pos_type="laplacian_eigval")
        graph_pe = np.real(graph_pe).astype(np.float32)
        node_pe = graph_to_node(graph_pe, self.num_nodes, {})
        self.assertEqual(node_pe.shape, np.ones((self.num_nodes, num_pos)).shape)


if __name__ == "__main__":
    ut.main()
