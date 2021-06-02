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
    MetricWrapper,
    Thresholder,
    pearsonr,
    spearmanr,
    mean_squared_error,
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


class test_MetricWrapper(ut.TestCase):
    

    def test_target_nan_mask(self):

        torch.random.manual_seed(42)

        for sz in [(100, ), (100, 1), (100, 10)]:

            err_msg = f"sz = {sz}"
        
            # Generate prediction and target matrices
            target = torch.rand(sz, dtype=torch.float32)
            preds = (0.5 * target) + (0.5 * torch.rand(sz, dtype=torch.float32))
            is_nan = torch.rand(sz) > 0.3
            target = (target > 0.5).to(torch.float32)
            target[is_nan] = float('nan')
            

            # Compute score with different ways of ignoring NaNs
            metric = MetricWrapper(metric="mse", target_nan_mask=None)
            score1 = metric(preds, target)
            self.assertTrue(torch.isnan(score1), msg=err_msg)

            # Replace NaNs by 0
            metric = MetricWrapper(metric="mse", target_nan_mask=0)
            score2 = metric(preds, target)
            
            this_target = target.clone()
            this_target[is_nan] = 0
            this_preds = preds.clone()
            self.assertAlmostEqual(score2, mean_squared_error(this_preds, this_target), msg=err_msg)

            # Replace NaNs by 1.5
            metric = MetricWrapper(metric="mse", target_nan_mask=1.5)
            score3 = metric(preds, target)
            
            this_target = target.clone()
            this_target[is_nan] = 1.5
            this_preds = preds.clone()
            self.assertAlmostEqual(score3, mean_squared_error(this_preds, this_target), msg=err_msg)

            # Flatten matrix and ignore NaNs
            metric = MetricWrapper(metric="mse", target_nan_mask="ignore-flatten")
            score4 = metric(preds, target)
            
            this_target = target.clone()[~is_nan]
            this_preds = preds.clone()[~is_nan]
            self.assertAlmostEqual(score4, mean_squared_error(this_preds, this_target), msg=err_msg)

            # Ignore NaNs in each column and average the score
            metric = MetricWrapper(metric="mse", target_nan_mask="ignore-mean-label")
            score5 = metric(preds, target)
            
            this_target = target.clone()
            this_preds = preds.clone()
            this_is_nan = is_nan.clone()
            if len(sz) == 1:
                this_target = target.unsqueeze(-1)
                this_preds = preds.unsqueeze(-1)
                this_is_nan = is_nan.unsqueeze(-1)
            
            this_target = [this_target[:, ii][~this_is_nan[:, ii]] for ii in range(this_target.shape[1])]
            this_preds = [this_preds[:, ii][~this_is_nan[:, ii]] for ii in range(this_preds.shape[1])]
            mse = []
            for ii in range(len(this_preds)):
                mse.append(mean_squared_error(this_preds[ii], this_target[ii]))
            mse = torch.mean(torch.stack(mse))
            self.assertAlmostEqual(score5.tolist(), mse.tolist(), msg=err_msg)


if __name__ == "__main__":
    ut.main()
