import unittest as ut
import torch
from torch.nn import BCELoss, MSELoss, L1Loss
from torchmetrics.functional import auroc, average_precision, precision, accuracy, recall, pearson_corrcoef, spearman_corrcoef, r2_score
from copy import deepcopy

from goli.ipu.ipu_metrics import BCELossIPU, MSELossIPU, L1LossIPU, auroc_ipu, average_precision_ipu, precision_ipu, accuracy_ipu, recall_ipu, pearson_ipu, r2_score_ipu


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
        preds = deepcopy(self.preds)[:, :4]
        target = deepcopy(self.target)[:, 0]
        t = deepcopy(target)

        target[t < 0.4] = 0
        target[(t >= 0.4) & (t < 0.6)] = 1
        target[(t >= 0.6) & (t < 0.8)] = 2
        target[(t >= 0.8)] = 3

        target_nan = deepcopy(target)
        target_nan[self.is_nan[:, 0]] = float("nan")
        target_nan_bin = deepcopy(target_nan)
        target_nan_bin[target_nan > 0] = 1

        # Micro accuracy binary
        score_true = accuracy(preds[:, 0], target.to(int)>0, average="micro")
        score_ipu = accuracy_ipu(preds[:, 0], target>0, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Accuracy binary is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Accuracy binary is different"
        )

        # Micro accuracy binary with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = accuracy(preds[:, 0][not_nan], target_nan_bin[not_nan].to(int), average="micro")
        score_ipu = accuracy_ipu(preds[:, 0], target_nan_bin, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Accuracy binary with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro Accuracy binary IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Accuracy with NaN is different"
        )

        # Micro accuracy
        score_true = accuracy(preds, target.to(int), average="micro")
        score_ipu = accuracy_ipu(preds, target, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Accuracy is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Accuracy is different"
        )

        # Micro accuracy with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = accuracy(preds[not_nan], target[not_nan].to(int), average="micro")
        score_ipu = accuracy_ipu(preds, target_nan, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Accuracy with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro Accuracy IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Accuracy with NaN is different"
        )

        # Macro accuracy
        score_true = accuracy(preds, target.to(int), average="macro", num_classes=4)
        score_ipu = accuracy_ipu(preds, target, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Accuracy is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro Accuracy is different"
        )

        # Macro accuracy with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = accuracy(preds[not_nan], target[not_nan].to(int), average="macro", num_classes=4)
        score_ipu = accuracy_ipu(preds, target_nan, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Accuracy with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Macro Accuracy IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro Accuracy with NaN is different"
        )

        # Weighted accuracy
        score_true = accuracy(preds, target.to(int), average="weighted", num_classes=4)
        score_ipu = accuracy_ipu(preds, target, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted Accuracy is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Weighted Accuracy is different"
        )

        # Weighted accuracy with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = accuracy(preds[not_nan], target[not_nan].to(int), average="weighted", num_classes=4)
        score_ipu = accuracy_ipu(preds, target_nan, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted Accuracy with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Weighted Accuracy IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Accuracy with NaN is different"
        )



    def test_recall(self):
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

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = recall(preds[not_nan], target[not_nan].to(int), num_classes=1)
        score_ipu = recall_ipu(preds, target_nan, num_classes=1 )
        self.assertFalse(score_true.isnan(), "Regular Average Recall with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular Average Recall IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Recall with NaN is different"
        )


    def test_pearsonr(self):
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0] + preds
        target_nan = deepcopy(target)
        target_nan[self.is_nan[:, 0]] = float("nan")


        # Regular loss
        score_true = pearson_corrcoef(preds, target)
        score_ipu = pearson_ipu(preds, target)
        self.assertFalse(score_true.isnan(), "Pearson is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=4, msg="Pearson is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = pearson_corrcoef(preds[not_nan], target[not_nan])
        score_ipu = pearson_ipu(preds, target_nan)
        self.assertFalse(score_true.isnan(), "Regular PearsonR with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "IPU PearsonR score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=4, msg="Pearson with NaN is different"
        )


    def test_r2_score(self):
        preds = deepcopy(self.preds)
        target = deepcopy(self.target) + preds
        target_nan = deepcopy(target)
        target_nan[self.is_nan] = float("nan")


        # Regular loss
        score_true = r2_score(preds, target)
        score_ipu = r2_score_ipu(preds, target)
        self.assertFalse(score_true.isnan(), "r2_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=4, msg="r2_score is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_ipu = r2_score_ipu(preds, target_nan, multioutput="raw_values")
        for ii in range(preds.shape[1]):
            score_true = r2_score(preds[:, ii][not_nan[:, ii]], target_nan[:, ii][not_nan[:, ii]], multioutput="raw_values")
            self.assertFalse(score_true.isnan().any(), f"{ii}: r2_score with target_nan is NaN")
            self.assertFalse(score_ipu[ii].isnan().any(), f"{ii}: IPU r2_score with target_nan is NaN")
            self.assertAlmostEqual(
                score_true.item(), score_ipu[ii].item(), places=4, msg=f"{ii}: r2_score with NaN is different"
            )

if __name__ == "__main__":
    ut.main()
