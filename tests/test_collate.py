"""
Unit tests for the collate
"""

import unittest as ut
from copy import deepcopy
import torch
import numpy as np

from goli.data.collate import collate_labels, goli_collate_fn


class test_Collate(ut.TestCase):
    def test_collate_labels(self):
        # Create fake labels
        labels_size_dict = {"label1": [1], "label2": [5], "label3": [5, 2]}
        fake_label = {
            "label1": torch.FloatTensor([1]),
            "label2": torch.FloatTensor([1, 2, 3, 4, 5]),
            "label3": torch.FloatTensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
        }
        fake_labels = []
        num_labels = 10
        for i in range(num_labels):
            fake_labels.append(deepcopy({key: val + 17 * i for key, val in fake_label.items()}))

        # Collate labels and check for the right shapes
        collated_labels = collate_labels(deepcopy(fake_labels), labels_size_dict)
        self.assertEqual(collated_labels["label1"].y.shape, torch.Size([num_labels, 1]))  # , 1
        self.assertEqual(collated_labels["label2"].y.shape, torch.Size([num_labels * 5, 1]))  # , 5
        self.assertEqual(collated_labels["label3"].y.shape, torch.Size([num_labels * 5, 2]))  # , 5, 2

        # Check that the values are correct
        label1_true = deepcopy(torch.stack([this_label["label1"] for this_label in fake_labels]))
        label2_true = deepcopy(torch.stack([this_label["label2"] for this_label in fake_labels]))
        label3_true = deepcopy(torch.stack([this_label["label3"] for this_label in fake_labels]))

        # NOTE: Flatten due to the way Data objects are collated (concat along first dim, instead of stacked)
        np.testing.assert_array_equal(collated_labels["label1"].y.numpy(), label1_true.numpy())
        np.testing.assert_array_equal(
            collated_labels["label2"].y.numpy(), label2_true.flatten(0, 1).unsqueeze(1).numpy()
        )
        np.testing.assert_array_equal(collated_labels["label3"].y.numpy(), label3_true.flatten(0, 1).numpy())

        # Remove some labels and check that the collation still works and puts `nan` in the right places
        missing_labels = {"label1": [1, 3, 5, 7, 9], "label2": [0, 2, 4, 6, 8], "label3": [0, 1, 2, 3, 4]}
        for key, missing_idx in missing_labels.items():
            for idx in missing_idx:
                fake_labels[idx].pop(key)
        label1_true[missing_labels["label1"]] = float("nan")
        label2_true[missing_labels["label2"]] = float("nan")
        label3_true[missing_labels["label3"]] = float("nan")

        # Collate labels and check for the right shapes
        collated_labels = collate_labels(deepcopy(fake_labels), labels_size_dict)
        self.assertEqual(collated_labels["label1"].y.shape, torch.Size([num_labels, 1]))  # , 1
        self.assertEqual(collated_labels["label2"].y.shape, torch.Size([num_labels * 5, 1]))  # , 5
        self.assertEqual(collated_labels["label3"].y.shape, torch.Size([num_labels * 5, 2]))  # , 5, 2

        # Check that the values are correct when some labels are missing
        # NOTE: Flatten due to the way Data objects are collated (concat along first dim, instead of stacked)
        np.testing.assert_array_equal(collated_labels["label1"].y.numpy(), label1_true.numpy())
        np.testing.assert_array_equal(
            collated_labels["label2"].y.numpy(), label2_true.flatten(0, 1).unsqueeze(1).numpy()
        )
        np.testing.assert_array_equal(collated_labels["label3"].y.numpy(), label3_true.flatten(0, 1).numpy())

        # Now test the `goli_collate_fn` function when only labels are given
        fake_labels2 = [{"labels": this_label} for this_label in fake_labels]
        collated_labels = goli_collate_fn(deepcopy(fake_labels2), labels_size_dict=labels_size_dict)["labels"]
        self.assertEqual(collated_labels["label1"].y.shape, torch.Size([num_labels, 1]))
        self.assertEqual(collated_labels["label2"].y.shape, torch.Size([num_labels * 5, 1]))  # , 5
        self.assertEqual(collated_labels["label3"].y.shape, torch.Size([num_labels * 5, 2]))  # , 5, 2

        # Check that the values are correct when some labels are missing
        # NOTE: Flatten due to the way Data objects are collated (concat along first dim, instead of stacked)
        np.testing.assert_array_equal(collated_labels["label1"].y.numpy(), label1_true.numpy())
        np.testing.assert_array_equal(
            collated_labels["label2"].y.numpy(), label2_true.flatten(0, 1).unsqueeze(1).numpy()
        )
        np.testing.assert_array_equal(collated_labels["label3"].y.numpy(), label3_true.flatten(0, 1).numpy())


if __name__ == "__main__":
    ut.main()
