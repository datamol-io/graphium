"""
Unit tests for the collate
"""

import unittest as ut
from copy import deepcopy
import torch
import numpy as np
from torch_geometric.data import Data

from graphium.data.collate import collate_labels, graphium_collate_fn


class test_Collate(ut.TestCase):
    def test_collate_labels(self):
        # Create fake labels
        labels_size_dict = {
            "graph_label1": [1],
            "graph_label2": [3],
            "node_label2": [5],
            "edge_label3": [5, 2],
            "node_label4": [5, 1],
        }
        labels_dtype_dict = {
            "graph_label1": torch.float32,
            "graph_label2": torch.float16,
            "node_label2": torch.float32,
            "edge_label3": torch.float32,
            "node_label4": torch.float32,
        }
        fake_label = {
            "graph_label1": torch.FloatTensor([1]),
            "graph_label2": torch.HalfTensor([1, 2, 3]),
            "node_label2": torch.FloatTensor([1, 2, 3, 4, 5]),
            "edge_label3": torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
            "node_label4": torch.FloatTensor([[1], [2], [3], [4], [5]]),
        }
        fake_labels = []
        num_labels = 10
        for i in range(num_labels):
            pyg_labels = Data(x=torch.empty(5, 0), edge_index=torch.empty(2, 5))
            for key, val in fake_label.items():
                pyg_labels[key] = val + 17 * 2
            fake_labels.append(pyg_labels)

        # Collate labels and check for the right shapes and dtypes
        collated_labels = collate_labels(
            deepcopy(fake_labels), deepcopy(labels_size_dict), deepcopy(labels_dtype_dict)
        )
        self.assertEqual(collated_labels["graph_label1"].shape, torch.Size([num_labels, 1]))  # , 1
        self.assertEqual(collated_labels["graph_label2"].shape, torch.Size([num_labels, 3]))  # , 1
        self.assertEqual(collated_labels["node_label2"].shape, torch.Size([num_labels * 5, 1]))  # , 5
        self.assertEqual(collated_labels["edge_label3"].shape, torch.Size([num_labels * 5, 2]))  # , 5, 2
        self.assertEqual(collated_labels["node_label4"].shape, torch.Size([num_labels * 5, 1]))  # , 5, 1

        self.assertEqual(collated_labels["graph_label1"].dtype, torch.float32)
        self.assertEqual(collated_labels["graph_label2"].dtype, torch.float16)

        # Check that the values are correct
        graph_label1_true = deepcopy(torch.stack([this_label["graph_label1"] for this_label in fake_labels]))
        graph_label2_true = deepcopy(torch.stack([this_label["graph_label2"] for this_label in fake_labels]))
        label2_true = deepcopy(torch.stack([this_label["node_label2"] for this_label in fake_labels]))
        label3_true = deepcopy(torch.stack([this_label["edge_label3"] for this_label in fake_labels]))
        label4_true = deepcopy(torch.stack([this_label["node_label4"] for this_label in fake_labels]))

        # NOTE: Flatten due to the way Data objects are collated (concat along first dim, instead of stacked)
        np.testing.assert_array_equal(collated_labels["graph_label1"].numpy(), graph_label1_true.numpy())
        np.testing.assert_array_equal(collated_labels["graph_label2"].numpy(), graph_label2_true.numpy())
        np.testing.assert_array_equal(
            collated_labels["node_label2"].numpy(), label2_true.flatten(0, 1).unsqueeze(1).numpy()
        )
        np.testing.assert_array_equal(
            collated_labels["edge_label3"].numpy(), label3_true.flatten(0, 1).numpy()
        )
        np.testing.assert_array_equal(
            collated_labels["node_label4"].numpy(), label4_true.flatten(0, 1).numpy()
        )

        # Remove some labels and check that the collation still works and puts `nan` in the right places
        missing_labels = {
            "graph_label1": [1, 3, 5, 7, 9],
            "graph_label2": [0, 4, 3, 1, 7],
            "node_label2": [0, 2, 4, 6, 8],
            "edge_label3": [0, 1, 2, 3, 4],
            "node_label4": [5, 1, 4, 9, 6],
        }
        for key, missing_idx in missing_labels.items():
            for idx in missing_idx:
                fake_labels[idx].pop(key)
        graph_label1_true[missing_labels["graph_label1"]] = float("nan")
        graph_label2_true[missing_labels["graph_label2"]] = float("nan")
        label2_true[missing_labels["node_label2"]] = float("nan")
        label3_true[missing_labels["edge_label3"]] = float("nan")
        label4_true[missing_labels["node_label4"]] = float("nan")

        # Collate labels and check for the right shapes
        labels_size_dict = {
            "graph_label1": [1],
            "graph_label2": [3],
            "node_label2": [5],
            "edge_label3": [5, 2],
            "node_label4": [5, 1],
        }
        collated_labels = collate_labels(
            deepcopy(fake_labels), deepcopy(labels_size_dict), deepcopy(labels_dtype_dict)
        )
        self.assertEqual(collated_labels["graph_label1"].shape, torch.Size([num_labels, 1]))  # , 1
        self.assertEqual(collated_labels["graph_label2"].shape, torch.Size([num_labels, 3]))  # , 1
        self.assertEqual(collated_labels["node_label2"].shape, torch.Size([num_labels * 5, 1]))  # , 5
        self.assertEqual(collated_labels["edge_label3"].shape, torch.Size([num_labels * 5, 2]))  # , 5, 2
        self.assertEqual(collated_labels["node_label4"].shape, torch.Size([num_labels * 5, 1]))  # , 5, 1

        # Check that the values are correct when some labels are missing
        # NOTE: Flatten due to the way Data objects are collated (concat along first dim, instead of stacked)
        np.testing.assert_array_equal(collated_labels["graph_label1"].numpy(), graph_label1_true.numpy())
        np.testing.assert_array_equal(collated_labels["graph_label2"].numpy(), graph_label2_true.numpy())
        np.testing.assert_array_equal(
            collated_labels["node_label2"].numpy(), label2_true.flatten(0, 1).unsqueeze(1).numpy()
        )
        np.testing.assert_array_equal(
            collated_labels["edge_label3"].numpy(), label3_true.flatten(0, 1).numpy()
        )
        np.testing.assert_array_equal(
            collated_labels["node_label4"].numpy(), label4_true.flatten(0, 1).numpy()
        )
        # Now test the `graphium_collate_fn` function when only labels are given
        fake_labels2 = [{"labels": this_label} for this_label in fake_labels]
        collated_labels = graphium_collate_fn(
            deepcopy(fake_labels2), labels_size_dict=labels_size_dict, labels_dtype_dict=labels_dtype_dict
        )["labels"]
        self.assertEqual(collated_labels["graph_label1"].shape, torch.Size([num_labels, 1]))
        self.assertEqual(collated_labels["graph_label2"].shape, torch.Size([num_labels, 3]))
        self.assertEqual(collated_labels["node_label2"].shape, torch.Size([num_labels * 5, 1]))  # , 5
        self.assertEqual(collated_labels["edge_label3"].shape, torch.Size([num_labels * 5, 2]))  # , 5, 2
        self.assertEqual(collated_labels["node_label4"].shape, torch.Size([num_labels * 5, 1]))  # , 5, 1

        # Check that the values are correct when some labels are missing
        # NOTE: Flatten due to the way Data objects are collated (concat along first dim, instead of stacked)
        np.testing.assert_array_equal(collated_labels["graph_label1"].numpy(), graph_label1_true.numpy())
        np.testing.assert_array_equal(collated_labels["graph_label2"].numpy(), graph_label2_true.numpy())
        np.testing.assert_array_equal(
            collated_labels["node_label2"].numpy(), label2_true.flatten(0, 1).unsqueeze(1).numpy()
        )
        np.testing.assert_array_equal(
            collated_labels["edge_label3"].numpy(), label3_true.flatten(0, 1).numpy()
        )
        np.testing.assert_array_equal(
            collated_labels["node_label4"].numpy(), label4_true.flatten(0, 1).numpy()
        )


if __name__ == "__main__":
    ut.main()
