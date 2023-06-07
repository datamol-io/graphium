# General imports
import unittest as ut
import numpy as np

import torch
from torch_geometric.data import Data, Batch

# Current library imports
from graphium.utils.packing import (
    smart_packing,
    get_pack_sizes,
    fast_packing,
    hybrid_packing,
    node_to_pack_indices_mask,
)


def random_packing(num_nodes, batch_size):
    ipu_batch_size = int(len(num_nodes) / batch_size)
    indices = np.arange(len(num_nodes))
    np.random.shuffle(indices)
    indices = np.reshape(indices, (ipu_batch_size, batch_size)).tolist()
    return indices


class test_Packing(ut.TestCase):
    def test_smart_packing(self):
        np.random.seed(42)

        batch_sizes = [2, 4, 8, 16, 32, 64]
        ipu_batch_sizes = [2, 3, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:
                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = smart_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_fast_packing(self):
        np.random.seed(42)

        # Start at 4 for fast_packing for better statistical significance
        batch_sizes = [4, 8, 16, 32, 64]
        ipu_batch_sizes = [4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:
                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = fast_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_hybrid_packing(self):
        np.random.seed(42)

        batch_sizes = [2, 4, 8, 16, 32, 64]
        ipu_batch_sizes = [2, 3, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:
                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = hybrid_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_node_to_pack_indices_mask(self):
        # Create a dummy batch
        in_dim = 7
        in_dim_edges = 11
        max_num_nodes_per_graph = 20
        batch_size_per_pack = 5

        torch.manual_seed(42)

        # Create a dummy batch of graphs
        batch, all_num_nodes = [], []
        for ii in range(100):
            num_nodes = torch.randint(1, max_num_nodes_per_graph, (1,)).item()
            all_num_nodes.append(num_nodes)
            num_edges = abs(round(2.2 * num_nodes) + torch.randint(-2, 2, (1,)).item()) + 1
            x = torch.randn(num_nodes, in_dim, dtype=torch.float32)
            edge_idx = torch.randint(0, num_nodes, (2, num_edges))
            e = torch.randn(edge_idx.shape[-1], in_dim_edges, dtype=torch.float32)
            g = Data(h=x, edge_index=edge_idx, edge_attr=e)
            batch.append(g)
        batch = Batch.from_data_list(batch)

        # Get the packing
        packed_graph_idx = fast_packing(all_num_nodes, batch_size_per_pack)
        pack_sizes = get_pack_sizes(packed_graph_idx, all_num_nodes)
        max_pack_size = max(pack_sizes)
        num_packs = len(pack_sizes)

        # Get the node to pack indices and the mask
        pack_from_node_idx, pack_attn_mask = node_to_pack_indices_mask(packed_graph_idx, all_num_nodes)

        # Assert that the nodes to pack indices are correct
        h = torch.arange(batch.num_nodes, dtype=torch.float32)
        packed_shape = [num_packs, max_pack_size]
        h_packed = torch.zeros(packed_shape)
        h_packed[pack_from_node_idx[:, 0], pack_from_node_idx[:, 1]] = h
        h_packed_unique = torch.sort(torch.unique(h_packed))[0]
        np.testing.assert_array_equal(h_packed_unique, torch.arange(batch.num_nodes))
        self.assertEqual(h_packed.sum(), h.sum())

        # Test again with additional h dimension
        h = batch.h
        packed_shape = [num_packs, max_pack_size] + list(h.shape[1:])
        h_packed = torch.zeros(packed_shape)
        h_packed[pack_from_node_idx[:, 0], pack_from_node_idx[:, 1]] = h
        h_packed_unique = torch.sort(torch.unique(h_packed))[0]
        h_packed_unique = h_packed_unique[h_packed_unique != 0]
        np.testing.assert_array_almost_equal(h_packed_unique, torch.unique(h))
        self.assertAlmostEqual(h_packed.sum().item(), h.sum().item(), places=3)

        # Assert that the mask is correct by counting the number of False values (the sum of squared number of nodes per pack)
        num_false = (~pack_attn_mask).sum([1, 2])
        num_expected = torch.as_tensor(
            [sum([all_num_nodes[graph_idx] ** 2 for graph_idx in pack]) for pack in packed_graph_idx]
        )
        np.testing.assert_array_equal(num_false, num_expected)

        # Assert that the mask is correct by counting the number of elements in each row and column
        num_expected = []
        for pack in packed_graph_idx:
            pack_num_expected = []
            for graph_idx in pack:
                num_nodes = all_num_nodes[graph_idx]
                for ii in range(num_nodes):
                    pack_num_expected.append(num_nodes)
            pack_num_expected.extend([0] * (max_pack_size - len(pack_num_expected)))
            num_expected.append(pack_num_expected)
        num_expected = torch.as_tensor(num_expected)
        num_false_row = (~pack_attn_mask).sum([2])
        num_false_col = (~pack_attn_mask).sum([1])
        np.testing.assert_array_equal(num_false_row, num_expected)
        np.testing.assert_array_equal(num_false_col, num_expected)


if __name__ == "__main__":
    ut.main()
