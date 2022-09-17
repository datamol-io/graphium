import unittest as ut
import numpy as np

from goli.ipu.ipu_dataloader import smart_packing, get_pack_sizes


def random_packing(num_nodes, batch_size):
    ipu_batch_size = int(len(num_nodes) / batch_size)
    indices = np.arange(len(num_nodes))
    np.random.shuffle(indices)
    indices = np.reshape(indices, (ipu_batch_size, batch_size)).tolist()
    return indices


class test_SmartPacking(ut.TestCase):
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

                # Assert that the smart packing is better than the random packing (when bz big enough for randomness)
                if (batch_size >= 4) and (ipu_batch_size >= 4):
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


if __name__ == "__main__":
    ut.main()
