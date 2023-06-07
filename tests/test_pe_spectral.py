"""
Unit tests for the positional encodings in graphium/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

from graphium.features.spectral import compute_laplacian_pe


class test_pe_spectral(ut.TestCase):
    # 2 disconnected 3 cliques
    adj1 = np.zeros((6, 6))
    adj_3clq = 1 - np.eye(3)
    adj1[:3, :3] = adj_3clq
    adj1[3:, 3:] = adj_3clq

    # 3-clique
    adj2 = 1 - np.eye(6)

    def test_for_connected_vs_disconnected_graph(self):
        num_pos = 3

        # test if pe works identically on connected vs disconnected graphs
        eigvals_pe1, _, _, cache = compute_laplacian_pe(self.adj1, num_pos, cache={})
        eigvals_pe1 = np.real(eigvals_pe1).astype(np.float32)
        _, eigvecs_pe1, _, _ = compute_laplacian_pe(self.adj1, num_pos, cache=cache)

        # We expect to cache 4 objects in when running the functon for the first time
        self.assertEqual(len(cache.keys()), 4)

        eigvals_pe2, _, _, _ = compute_laplacian_pe(self.adj2, num_pos, cache={})
        eigvals_pe2 = np.real(eigvals_pe2).astype(np.float32)
        _, eigvecs_pe2, _, _ = compute_laplacian_pe(self.adj2, num_pos, cache={})

        np.testing.assert_array_almost_equal(2 * eigvals_pe1, eigvals_pe2)
        self.assertListEqual(list(eigvals_pe2.shape), [self.adj2.shape[0], num_pos])
        self.assertListEqual(list(eigvecs_pe2.shape), [self.adj2.shape[0], num_pos])


if __name__ == "__main__":
    ut.main()
