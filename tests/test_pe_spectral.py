"""
Unit tests for the positional encodings in goli/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

from goli.features.spectral import compute_laplacian_pe


class test_pe_spectral(ut.TestCase):

    # 2 disconnected 3 cliques
    adj1 = np.zeros((6,6))
    adj_3clq = 1 - np.eye(3)
    adj1[:3,:3] = adj_3clq
    adj1[3:,3:] = adj_3clq

    # 3-clique
    adj2 = 1 - np.eye(3)


    def test_for_connected_vs_disconnected_graph(self):
        num_pos = 3

        # test if pe works identically on connected vs disconnected graphs
        graph_pe1, _, cache = compute_laplacian_pe(self.adj1, num_pos, cache={}, pos_type="laplacian_eigval")
        node_pe1, _, _ = compute_laplacian_pe(self.adj1, num_pos, cache=cache, pos_type="laplacian_eigvec")
        graph_pe1 = np.real(graph_pe1).astype(np.float32)

        # We expect to cache 4 objects in when running the functon for the first time
        self.assertEqual(len(cache.keys()), 4)
        
        graph_pe2, _, _ = compute_laplacian_pe(self.adj2, num_pos, cache={}, pos_type="laplacian_eigval")
        node_pe2, _, _ = compute_laplacian_pe(self.adj2, num_pos, cache={}, pos_type="laplacian_eigvec")
        graph_pe2 = np.real(graph_pe2).astype(np.float32)

        for i in range(2):
            np.testing.assert_array_almost_equal(graph_pe1[i], graph_pe2)
            np.testing.assert_array_almost_equal(node_pe1[i], node_pe2)
            self.assertEqual(graph_pe2.shape, np.zeros(num_pos).shape)
            self.assertEqual(node_pe2.shape, np.zeros((self.adj2.shape[0], num_pos)).shape)


if __name__ == "__main__":
    ut.main()