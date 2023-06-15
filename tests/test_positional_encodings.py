"""
Unit tests for the positional encodings in graphium/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

# from graphium.features.spectral import compute_laplacian_positional_eigvecs # TODO: add tests
# from graphium.features.rw import compute_rwse # TODO: add tests
from graphium.features.electrostatic import compute_electrostatic_interactions
from graphium.features.commute import compute_commute_distances
from graphium.features.graphormer import compute_graphormer_distances


class test_positional_encodings(ut.TestCase):
    # Test graphs
    adj_dict = {}
    max_dict = {}

    # 6-ring
    adj = np.asarray(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ]
    )
    adj_dict["6-ring"] = adj
    max_dict["6-ring"] = 3

    # 5-path
    G = nx.path_graph(5)
    adj = nx.to_numpy_array(G)
    adj_dict["5-path"] = adj
    max_dict["5-path"] = 4

    # 4-clique
    adj = 1 - np.eye(4)
    adj_dict["4-clique"] = adj
    max_dict["4-clique"] = 1

    # 4-barbell
    H = nx.barbell_graph(4, 0)
    adj = nx.to_numpy_array(H)
    adj_dict["4-barbell"] = adj
    max_dict["4-barbell"] = 3

    def test_dimensions(self):
        for _, adj in self.adj_dict.items():
            pe, _, _ = compute_electrostatic_interactions(adj, cache={})
            self.assertEqual(pe.shape, adj.shape)

            pe, _, _ = compute_graphormer_distances(adj, adj.shape[0], cache={})
            self.assertEqual(pe.shape, adj.shape)

            pe, _, _ = compute_commute_distances(adj, adj.shape[0], cache={})
            self.assertEqual(pe.shape, adj.shape)

    def test_symmetry(self):
        for _, adj in self.adj_dict.items():
            pe, _, _ = compute_graphormer_distances(adj, adj.shape[0], cache={})
            np.testing.assert_array_almost_equal(pe, pe.T)

            pe, _, _ = compute_commute_distances(adj, adj.shape[0], cache={})
            np.testing.assert_array_almost_equal(pe, pe.T)

    def test_max_dist(self):
        for key, adj in self.adj_dict.items():
            pe, _, _ = compute_graphormer_distances(adj, adj.shape[0], cache={})
            np.testing.assert_array_almost_equal(pe.max(), self.max_dict[key])


if __name__ == "__main__":
    ut.main()
