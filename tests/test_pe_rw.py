"""
Unit tests for the positional encodings in graphium/features/*
"""

import numpy as np
import networkx as nx
import unittest as ut

from graphium.features.rw import compute_rwse


class test_pe_spectral(ut.TestCase):
    def test_caching_and_outputs(self):
        # 4-barbell
        G = nx.barbell_graph(4, 0)
        adj = nx.to_numpy_array(G)
        num_nodes = adj.shape[0]
        cache = {}

        ksteps1 = [4, 6]
        ksteps2 = [2]
        ksteps3 = [6, 7]

        pe1, _, cache = compute_rwse(
            adj.astype(np.float32), ksteps1, num_nodes, cache, pos_type="rw_transition_probs"
        )

        pe2, _, cache = compute_rwse(
            adj.astype(np.float32), ksteps2, num_nodes, cache, pos_type="rw_return_probs"
        )

        pe3, _, cache = compute_rwse(
            adj.astype(np.float32), ksteps3, num_nodes, cache, pos_type="rw_return_probs"
        )

        self.assertTrue(all([k in cache["ksteps"] for k in ksteps1 + ksteps2 + ksteps3]))
        self.assertTrue(pe1.shape, np.zeros((num_nodes, num_nodes, len(ksteps1))))
        self.assertTrue(pe2.shape, np.zeros((num_nodes, len(ksteps2))))
        self.assertTrue(pe3.shape, np.zeros((num_nodes, len(ksteps3))))

        for i in range(len(ksteps1)):
            np.testing.assert_array_almost_equal(pe1[..., i].sum(1), np.ones(num_nodes))

        self.assertGreaterEqual(pe1.min(), 0.0)
        self.assertLessEqual(pe1.max(), 1.0)

        self.assertGreaterEqual(pe2.min(), 0.0)
        self.assertLessEqual(pe2.max(), 1.0)

        self.assertGreaterEqual(pe3.min(), 0.0)
        self.assertLessEqual(pe3.max(), 1.0)


if __name__ == "__main__":
    ut.main()
