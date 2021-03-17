"""
Unit tests for the different layers of goli/dgl/dgl_layers/...

The layers are not thoroughly tested due to the difficulty of testing them
"""

import numpy as np
import torch
import unittest as ut
import dgl
from copy import deepcopy
import inspect

from goli.trainer.metrics import (
    METRICS_REGRESSION,
    METRICS_CLASSIFICATION,
    MetricWithThreshold,
    Thresholder,
    pearsonr,
    spearmanr,
)


class test_Metrics(ut.TestCase):
    def test_thresholder(self):

        torch.manual_seed(42)
        preds = torch.rand(100, dtype=torch.float32)
        target = torch.rand(100, dtype=torch.float32)

        th = 0.7
        preds_greater = preds > th
        target_greater = target > th

        # Test thresholder greater
        for th_on_preds in [True, False]:
            for th_on_target in [True, False]:
                thresholder = Thresholder(
                    threshold=th, operator="greater", th_on_target=th_on_target, th_on_preds=th_on_preds
                )
                preds2, target2 = thresholder(preds, target)
                if th_on_preds:
                    self.assertListEqual(preds2.tolist(), preds_greater.tolist())
                else:
                    self.assertListEqual(preds2.tolist(), preds.tolist())
                if th_on_target:
                    self.assertListEqual(target2.tolist(), target_greater.tolist())
                else:
                    self.assertListEqual(target2.tolist(), target.tolist())

        # Test thresholder lower
        for th_on_preds in [True, False]:
            for th_on_target in [True, False]:
                thresholder = Thresholder(
                    threshold=th, operator="lower", th_on_target=th_on_target, th_on_preds=th_on_preds
                )
                preds2, target2 = thresholder(preds, target)
                if th_on_preds:
                    self.assertListEqual(preds2.tolist(), (~preds_greater).tolist())
                else:
                    self.assertListEqual(preds2.tolist(), preds.tolist())
                if th_on_target:
                    self.assertListEqual(target2.tolist(), (~target_greater).tolist())
                else:
                    self.assertListEqual(target2.tolist(), target.tolist())

    def test_pearsonr_spearmanr(self):
        preds = torch.tensor([0.0, 1, 2, 3])
        target = torch.tensor([0.0, 1, 2, 1.5])

        self.assertAlmostEqual(pearsonr(preds, target).tolist(), 0.8315, places=4)
        self.assertAlmostEqual(spearmanr(preds, target).tolist(), 0.8, places=4)

        preds = torch.tensor([76, 25, 72, 0, 60, 96, 55, 57, 10, 26, 47, 87, 97, 2, 20])
        target = torch.tensor([12, 80, 35, 6, 58, 22, 41, 66, 92, 55, 46, 61, 89, 83, 14])

        self.assertAlmostEqual(pearsonr(preds, target).tolist(), -0.0784, places=4)
        self.assertAlmostEqual(spearmanr(preds, target).tolist(), -0.024999, places=4)

        preds = preds.repeat(2, 1).T
        target = target.repeat(2, 1).T

        self.assertAlmostEqual(pearsonr(preds, target).tolist(), -0.0784, places=4)
        self.assertAlmostEqual(spearmanr(preds, target).tolist(), -0.024999, places=4)


if __name__ == "__main__":
    ut.main()
