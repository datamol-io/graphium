import unittest as ut
import torch
from torch.nn import BCELoss, MSELoss, L1Loss
from copy import deepcopy

from graphium.ipu.ipu_losses import BCELossIPU, MSELossIPU, L1LossIPU


class test_Losses(ut.TestCase):
    torch.manual_seed(42)
    preds = torch.rand((100, 10), dtype=torch.float32)
    target = torch.rand((100, 10), dtype=torch.float32)

    th = 0.7
    nan_th = 0.2
    preds_greater = preds > th
    target_greater = (target > th).to(torch.float32)
    target_greater_nan = deepcopy(target_greater)
    is_nan = target < nan_th
    target_greater_nan[target < nan_th] = torch.nan
    target_nan = deepcopy(target)
    target_nan[target < nan_th] = torch.nan

    def test_bce(self):
        preds = deepcopy(self.preds)
        target = deepcopy(self.target_greater)
        target_nan = deepcopy(self.target_greater_nan)

        # Regular loss
        loss_true = BCELoss()(preds, target)
        loss_ipu = BCELossIPU()(preds, target)
        self.assertFalse(loss_true.isnan(), "Regular BCELoss is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular BCELoss is different"
        )

        # Weighted loss
        weight = torch.rand(preds.shape[1], dtype=torch.float32)
        loss_true = BCELoss(weight=weight)(preds, target)
        loss_ipu = BCELossIPU(weight=weight)(preds, target)
        self.assertFalse(loss_true.isnan(), "Regular BCELoss is NaN")
        self.assertAlmostEqual(loss_true.item(), loss_ipu.item(), msg="Weighted BCELoss is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = BCELoss()(preds[not_nan], target[not_nan])
        loss_ipu = BCELossIPU()(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Regular BCELoss with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular BCELossIPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular BCELoss with NaN is different"
        )

        # Weighted loss with NaNs in target
        not_nan = ~target_nan.isnan()
        weight = torch.rand(preds.shape, dtype=torch.float32)
        loss_true = BCELoss(weight=weight[not_nan])(preds[not_nan], target_nan[not_nan])
        loss_ipu = BCELossIPU(weight=weight)(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Weighted BCELoss with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Weighted BCELossIPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Weighted BCELoss with NaN is different"
        )

    def test_mse(self):
        preds = deepcopy(self.preds)
        target = deepcopy(self.target)
        target_nan = deepcopy(self.target_nan)

        # Regular loss
        loss_true = MSELoss()(preds, target)
        loss_ipu = MSELossIPU()(preds, target)
        self.assertFalse(loss_true.isnan(), "Regular MSELoss is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular MSELoss is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = MSELoss()(preds[not_nan], target[not_nan])
        loss_ipu = MSELossIPU()(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Regular MSELoss with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular MSELossIPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular MSELoss with NaN is different"
        )

    def test_l1(self):
        preds = deepcopy(self.preds)
        target = deepcopy(self.target)
        target_nan = deepcopy(self.target_nan)

        # Regular loss
        loss_true = L1Loss()(preds, target)
        loss_ipu = L1LossIPU()(preds, target)
        self.assertFalse(loss_true.isnan(), "Regular MAELoss is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular MAELoss is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = L1Loss()(preds[not_nan], target[not_nan])
        loss_ipu = L1LossIPU()(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Regular MAELoss with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular MAELossIPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular MAELoss with NaN is different"
        )
