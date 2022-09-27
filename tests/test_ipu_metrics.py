import unittest as ut
import torch
from torch.nn import BCELoss, MSELoss, L1Loss
from torchmetrics.functional import auroc, average_precision, precision, accuracy, recall
from copy import deepcopy

from goli.ipu.ipu_metrics import BCELossIPU, MSELossIPU, L1LossIPU, auroc_ipu, average_precision_ipu, precision_ipu, accuracy_ipu, recall_ipu


class test_Losses(ut.TestCase):

    torch.manual_seed(42)
    preds = torch.rand((100, 10), dtype=torch.float32)
    target = torch.rand((100, 10), dtype=torch.float32)

    th = 0.7
    nan_th = 0.2
    preds_greater = preds > th
    target_greater = (target > th).to(torch.float32)
    target_greater_nan = deepcopy(target_greater)
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
        self.assertFalse(loss_true.isnan(), "Regular MSELoss is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular MSELoss is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = L1Loss()(preds[not_nan], target[not_nan])
        loss_ipu = L1LossIPU()(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Regular MSELoss with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular MSELossIPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular MSELoss with NaN is different"
        )

    def test_auroc(self):
        preds_with_weights = deepcopy(self.preds)
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = auroc(preds, target.to(int), num_classes=1)
        score_ipu = auroc_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular AUROC score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular AUROC score is different"
        )

        # Weighted loss (As in BCE)
        sample_weights = torch.rand(preds.shape[0], dtype=torch.float32)
        score_true = auroc(preds, target.to(int), num_classes=1, sample_weights=sample_weights)
        score_ipu = auroc_ipu(preds, target, num_classes=1, sample_weights=sample_weights)
        self.assertFalse(score_true.isnan(), "Regular AUROC score is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), msg="Weighted AUROC score is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = auroc(preds[not_nan], target[not_nan].to(int), num_classes=1)
        score_ipu = auroc_ipu(preds, target_nan, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular AUROC score with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular AUROCIPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular AUROC score with NaN is different"
        )

        # Weighted loss with NaNs in target (As in BCE)
        not_nan = ~target_nan.isnan()
        sample_weights = torch.rand(preds.shape, dtype=torch.float32)
        loss_true = auroc(preds[not_nan], target_nan[not_nan].to(int), sample_weights=sample_weights[not_nan])
        loss_ipu = auroc_ipu(preds, target_nan, sample_weights=sample_weights)
        self.assertFalse(loss_true.isnan(), "Weighted AUROC score with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Weighted AUROC IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            # AssertionError: 0.6603766679763794 != 0.6234951615333557 within 2 places
            loss_true.item(), loss_ipu.item(), places=6, msg="Weighted AUROC with NaN is different"
        )

    def test_average_precision(self):
        preds_with_weights = deepcopy(self.preds)
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = average_precision(preds, target.to(int), num_classes=1)
        score_ipu = average_precision_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Precision is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Precision is different"
        )

        # Weighted loss (As in BCE)
        sample_weights = torch.rand(preds.shape[0], dtype=torch.float32)
        score_true = average_precision(preds, target.to(int), num_classes=1, sample_weights=sample_weights)
        score_ipu = average_precision_ipu(preds, target, num_classes=1, sample_weights=sample_weights)
        self.assertFalse(score_true.isnan(), "Regular Average Precision is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), msg="Weighted Average Precision is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = average_precision(preds[not_nan], target[not_nan].to(int), num_classes=1)
        score_ipu = average_precision_ipu(preds, target_nan, num_classes=1 )
        self.assertFalse(score_true.isnan(), "Regular Average Precision with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular Average Precision IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Precision with NaN is different"
        )

        # Weighted loss with NaNs in target (As in BCE)
        not_nan = ~target_nan.isnan()
        sample_weights = torch.rand(preds.shape, dtype=torch.float32)
        loss_true = average_precision(preds[not_nan], target_nan[not_nan].to(int), sample_weights=sample_weights[not_nan])
        loss_ipu = average_precision_ipu(preds, target_nan, sample_weights=sample_weights)
        self.assertFalse(loss_true.isnan(), "Weighted Average Precision with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Weighted Average Precision IPU IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            # AssertionError: 0.6603766679763794 != 0.6234951615333557 within 2 places
            loss_true.item(), loss_ipu.item(), places=6, msg="Weighted Average Precision IPU with NaN is different"
        )

    def test_precision(self):
        preds_with_weights = deepcopy(self.preds)
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = precision(preds, target.to(int), num_classes=1)
        score_ipu = precision_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Precision is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Precision is different"
        )

        # Weighted loss (As in BCE)
        sample_weights = torch.rand(preds.shape[0], dtype=torch.float32)
        score_true = precision(preds, target.to(int), num_classes=1)
        score_ipu = precision_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Precision is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), msg="Weighted Average Precision is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = precision(preds[not_nan], target[not_nan].to(int), num_classes=1)
        score_ipu = precision_ipu(preds, target_nan, num_classes=1 )
        self.assertFalse(score_true.isnan(), "Regular Average Precision with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular Average Precision IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Precision with NaN is different"
        )

        # Weighted loss with NaNs in target (As in BCE)
        not_nan = ~target_nan.isnan()
        sample_weights = torch.rand(preds.shape, dtype=torch.float32)
        loss_true = precision(preds[not_nan], target_nan[not_nan].to(int))
        loss_ipu = precision_ipu(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Weighted Average Precision with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Weighted Average Precision IPU IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            # AssertionError: 0.6603766679763794 != 0.6234951615333557 within 2 places
            loss_true.item(), loss_ipu.item(), places=6, msg="Weighted Average Precision IPU with NaN is different"
        )

    def test_accuracy(self):
        preds_with_weights = deepcopy(self.preds)
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = accuracy(preds, target.to(int), num_classes=1)
        score_ipu = accuracy_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Accuracy is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Accuracy is different"
        )

        # Weighted loss (As in BCE)
        sample_weights = torch.rand(preds.shape[0], dtype=torch.float32)
        score_true = accuracy(preds, target.to(int), num_classes=1)
        score_ipu = accuracy_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Accuracy is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), msg="Weighted Average Accuracy is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = accuracy(preds[not_nan], target[not_nan].to(int), num_classes=1)
        score_ipu = accuracy_ipu(preds, target_nan, num_classes=1 )
        self.assertFalse(score_true.isnan(), "Regular Average Accuracy with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular Average Accuracy IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Accuracy with NaN is different"
        )

        # Weighted loss with NaNs in target (As in BCE)
        not_nan = ~target_nan.isnan()
        sample_weights = torch.rand(preds.shape, dtype=torch.float32)
        loss_true = accuracy(preds[not_nan], target_nan[not_nan].to(int))
        loss_ipu = accuracy_ipu(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Weighted Average Accuracy with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Weighted Average Accuracy IPU IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            # AssertionError: 0.6603766679763794 != 0.6234951615333557 within 2 places
            loss_true.item(), loss_ipu.item(), places=6, msg="Weighted Average Accuracy IPU with NaN is different"
        )

    def test_recall(self):
        preds_with_weights = deepcopy(self.preds)
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = recall(preds, target.to(int), num_classes=1)
        score_ipu = recall_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Recall is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Recall is different"
        )

        # Weighted loss (As in BCE)
        sample_weights = torch.rand(preds.shape[0], dtype=torch.float32)
        score_true = recall(preds, target.to(int), num_classes=1)
        score_ipu = recall_ipu(preds, target, num_classes=1)
        self.assertFalse(score_true.isnan(), "Regular Average Recall is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), msg="Weighted Average Recall is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = recall(preds[not_nan], target[not_nan].to(int), num_classes=1)
        score_ipu = recall_ipu(preds, target_nan, num_classes=1 )
        self.assertFalse(score_true.isnan(), "Regular Average Recall with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular Average Recall IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Recall with NaN is different"
        )

        # Weighted loss with NaNs in target (As in BCE)
        not_nan = ~target_nan.isnan()
        sample_weights = torch.rand(preds.shape, dtype=torch.float32)
        loss_true = recall(preds[not_nan], target_nan[not_nan].to(int))
        loss_ipu = recall_ipu(preds, target_nan)
        self.assertFalse(loss_true.isnan(), "Weighted Average Recall with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Weighted Average Recall IPU IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            # AssertionError: 0.6603766679763794 != 0.6234951615333557 within 2 places
            loss_true.item(), loss_ipu.item(), places=6, msg="Weighted Average Recall IPU with NaN is different"
        )

if __name__ == "__main__":
    ut.main()
