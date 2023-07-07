import unittest as ut
import torch
from torchmetrics.functional import (
    auroc,
    average_precision,
    precision,
    accuracy,
    recall,
    pearson_corrcoef,
    spearman_corrcoef,
    r2_score,
    f1_score,
    fbeta_score,
    mean_squared_error,
    mean_absolute_error,
)
from copy import deepcopy
import pytest

from graphium.ipu.ipu_metrics import (
    auroc_ipu,
    average_precision_ipu,
    precision_ipu,
    accuracy_ipu,
    recall_ipu,
    pearson_ipu,
    spearman_ipu,
    r2_score_ipu,
    f1_score_ipu,
    fbeta_score_ipu,
    mean_squared_error_ipu,
    mean_absolute_error_ipu,
)


@pytest.mark.ipu
class test_Metrics(ut.TestCase):
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

    def test_auroc(self):
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = auroc(preds, target.to(int))
        score_ipu = auroc_ipu(preds, target)
        self.assertFalse(score_true.isnan(), "Regular AUROC score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular AUROC score is different"
        )

        # Weighted loss (As in BCE)
        sample_weights = torch.rand(preds.shape[0], dtype=torch.float32)
        score_true = auroc(preds, target.to(int), sample_weights=sample_weights)
        score_ipu = auroc_ipu(preds, target, sample_weights=sample_weights)
        self.assertFalse(score_true.isnan(), "Regular AUROC score is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), msg="Weighted AUROC score is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = auroc(preds[not_nan], target[not_nan].to(int))
        score_ipu = auroc_ipu(preds, target_nan)
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
            loss_true.item(),
            loss_ipu.item(),
            places=6,
            msg="Weighted AUROC with NaN is different",
        )

    def test_average_precision(self):  # TODO: Make work with multi-class
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0]
        target_nan = deepcopy(self.target_nan)[:, 0]

        target[target < 0.5] = 0
        target[target >= 0.5] = 1

        target_nan[target_nan < 0.5] = 0
        target_nan[target_nan >= 0.5] = 1

        # Regular loss
        score_true = average_precision(preds, target.to(int), task="binary")
        score_ipu = average_precision_ipu(preds, target.to(int), task="binary")
        self.assertFalse(score_true.isnan(), "Regular Average Precision is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Regular Average Precision is different"
        )

        # Regular average precision with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = average_precision(preds[not_nan], target[not_nan].to(int), task="binary")
        score_ipu = average_precision_ipu(preds, target_nan, task="binary")
        self.assertFalse(score_true.isnan(), "Regular Average Precision with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Regular Average Precision IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Regular Average Precision with NaN is different",
        )

    def test_precision(self):
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

        # Micro precision binary
        score_true = precision(preds[:, 0], target.to(int) > 0, average="micro")
        score_ipu = precision_ipu(preds[:, 0], target > 0, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Precision binary is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Precision binary is different"
        )

        # Micro precision binary with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = precision(preds[:, 0][not_nan], target_nan_bin[not_nan].to(int), average="micro")
        score_ipu = precision_ipu(preds[:, 0], target_nan_bin, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Precision binary with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro Precision binary IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Precision with NaN is different"
        )

        # Micro precision
        score_true = precision(preds, target.to(int), average="micro")
        score_ipu = precision_ipu(preds, target, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Precision is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Precision is different"
        )

        # Micro precision with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = precision(preds[not_nan], target[not_nan].to(int), average="micro")
        score_ipu = precision_ipu(preds, target_nan, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Precision with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro Precision IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Precision with NaN is different"
        )

        # Macro precision
        score_true = precision(preds, target.to(int), average="macro", num_classes=4)
        score_ipu = precision_ipu(preds, target, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Precision is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro Precision is different"
        )

        # Macro precision multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = precision(preds[not_nan], target[not_nan].to(int), average="macro", num_classes=4)
        score_ipu = precision_ipu(preds, target_nan, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Precision multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Macro Precision multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Macro Precision multiclass with NaN is different",
        )

        # Macro precision multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = precision(preds[not_nan], target[not_nan].to(int), average="macro", num_classes=4)
        score_ipu = precision_ipu(preds, target_nan, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Precision multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Macro Precision multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Macro Precision multiclass with NaN is different",
        )

        # Weighted precision multiclass
        score_true = precision(preds, target.to(int), average="weighted", num_classes=4)
        score_ipu = precision_ipu(preds, target, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted Precision multiclass is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Weighted Precision multiclass is different"
        )

        # Weighted precision multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = precision(preds[not_nan], target[not_nan].to(int), average="weighted", num_classes=4)
        score_ipu = precision_ipu(preds, target_nan, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted Precision multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Weighted Precision multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Regular Average Precision multiclass with NaN is different",
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
        score_true = accuracy(preds[:, 0], target.to(int) > 0, average="micro")
        score_ipu = accuracy_ipu(preds[:, 0], target > 0, average="micro")
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
            score_true.item(), score_ipu.item(), places=6, msg="Regular Accuracy with NaN is different"
        )

    def test_recall(self):
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

        # Micro recall binary
        score_true = recall(preds[:, 0], target.to(int) > 0, average="micro")
        score_ipu = recall_ipu(preds[:, 0], target > 0, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Recall binary is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Recall binary is different"
        )

        # Micro recall binary with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = recall(preds[:, 0][not_nan], target_nan_bin[not_nan].to(int), average="micro")
        score_ipu = recall_ipu(preds[:, 0], target_nan_bin, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Recall binary with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro Recall binary IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Recall binary with NaN is different"
        )

        # Micro recall
        score_true = recall(preds, target.to(int), average="micro")
        score_ipu = recall_ipu(preds, target, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Recall is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), places=6, msg="Micro Recall is different")

        # Micro recall with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = recall(preds[not_nan], target[not_nan].to(int), average="micro")
        score_ipu = recall_ipu(preds, target_nan, average="micro")
        self.assertFalse(score_true.isnan(), "Micro Recall with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro Recall IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro Recall with NaN is different"
        )

        # Macro recall multiclass
        score_true = recall(preds, target.to(int), average="macro", num_classes=4)
        score_ipu = recall_ipu(preds, target, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Recall is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro Recall multiclass is different"
        )

        # Macro recall multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = recall(preds[not_nan], target[not_nan].to(int), average="macro", num_classes=4)
        score_ipu = recall_ipu(preds, target_nan, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro Recall multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Macro Recall multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro Recall multiclass with NaN is different"
        )

        # Weighted recallmulticlass
        score_true = recall(preds, target.to(int), average="weighted", num_classes=4)
        score_ipu = recall_ipu(preds, target, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted Recall is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Weighted Recall is different"
        )

        # Weighted recall multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = recall(preds[not_nan], target[not_nan].to(int), average="weighted", num_classes=4)
        score_ipu = recall_ipu(preds, target_nan, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted Recall multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Weighted Recall multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Regular Recall multiclass with NaN is different",
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
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), places=4, msg="Pearson is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = pearson_corrcoef(preds[not_nan], target[not_nan])
        score_ipu = pearson_ipu(preds, target_nan)
        self.assertFalse(score_true.isnan(), "Regular PearsonR with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "IPU PearsonR score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=4, msg="Pearson with NaN is different"
        )

    def test_spearmanr(self):
        preds = deepcopy(self.preds)[:, 0]
        target = deepcopy(self.target)[:, 0] + preds
        target_nan = deepcopy(target)
        target_nan[self.is_nan[:, 0]] = float("nan")

        # Regular loss
        score_true = spearman_corrcoef(preds, target)
        score_ipu = spearman_ipu(preds, target)
        self.assertFalse(score_true.isnan(), "Spearman is NaN")
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), places=4, msg="Spearman is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = spearman_corrcoef(preds[not_nan], target[not_nan])
        score_ipu = spearman_ipu(preds, target_nan)
        self.assertFalse(score_true.isnan(), "Regular Spearman with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "IPU Spearman score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=4, msg="Spearman with NaN is different"
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
        self.assertAlmostEqual(score_true.item(), score_ipu.item(), places=4, msg="r2_score is different")

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        score_ipu = r2_score_ipu(preds, target_nan, multioutput="raw_values")
        for ii in range(preds.shape[1]):
            score_true = r2_score(
                preds[:, ii][not_nan[:, ii]], target_nan[:, ii][not_nan[:, ii]], multioutput="raw_values"
            )
            self.assertFalse(score_true.isnan().any(), f"{ii}: r2_score with target_nan is NaN")
            self.assertFalse(score_ipu[ii].isnan().any(), f"{ii}: IPU r2_score with target_nan is NaN")
            self.assertAlmostEqual(
                score_true.item(), score_ipu[ii].item(), places=4, msg=f"{ii}: r2_score with NaN is different"
            )

    def test_fbeta_score(self):
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

        # Micro fbeta_score binary
        score_true = fbeta_score(preds[:, 0], target.to(int) > 0, average="micro", beta=0.5)
        score_ipu = fbeta_score_ipu(preds[:, 0], target > 0, average="micro", beta=0.5)
        self.assertFalse(score_true.isnan(), "Micro FBETA_score binary is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro FBETA_score binary is different"
        )

        # Micro fbeta_score binary with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = fbeta_score(
            preds[:, 0][not_nan], target_nan_bin[not_nan].to(int), average="micro", beta=0.5
        )
        score_ipu = fbeta_score_ipu(preds[:, 0], target_nan_bin, average="micro", beta=0.5)
        self.assertFalse(score_true.isnan(), "Micro FBETA_score binary with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro FBETA_score binary IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Micro FBETA_score binary with NaN is different",
        )

        # Micro fbeta_score
        score_true = fbeta_score(preds, target.to(int), average="micro", beta=0.5)
        score_ipu = fbeta_score_ipu(preds, target, average="micro", beta=0.5)
        self.assertFalse(score_true.isnan(), "Micro FBETA_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro FBETA_score is different"
        )

        # Micro fbeta_score with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = fbeta_score(preds[not_nan], target[not_nan].to(int), average="micro", beta=0.5)
        score_ipu = fbeta_score_ipu(preds, target_nan, average="micro", beta=0.5)
        self.assertFalse(score_true.isnan(), "Micro FBETA_score with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro FBETA_score IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro FBETA_score with NaN is different"
        )

        # Macro fbeta_score multiclass
        score_true = fbeta_score(preds, target.to(int), average="macro", num_classes=4, beta=0.5)
        score_ipu = fbeta_score_ipu(preds, target, average="macro", num_classes=4, beta=0.5)
        self.assertFalse(score_true.isnan(), "Macro FBETA_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro FBETA_score multiclass is different"
        )

        # Macro fbeta_score multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = fbeta_score(
            preds[not_nan], target[not_nan].to(int), average="macro", num_classes=4, beta=0.5
        )
        score_ipu = fbeta_score_ipu(preds, target_nan, average="macro", num_classes=4, beta=0.5)
        self.assertFalse(score_true.isnan(), "Macro FBETA_score multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Macro FBETA_score multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Macro FBETA_score multiclass with NaN is different",
        )

        # Weighted fbeta_scoremulticlass
        score_true = fbeta_score(preds, target.to(int), average="weighted", num_classes=4, beta=0.5)
        score_ipu = fbeta_score_ipu(preds, target, average="weighted", num_classes=4, beta=0.5)
        self.assertFalse(score_true.isnan(), "Weighted FBETA_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Weighted FBETA_score is different"
        )

        # Weighted fbeta_score multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = fbeta_score(
            preds[not_nan], target[not_nan].to(int), average="weighted", num_classes=4, beta=0.5
        )
        score_ipu = fbeta_score_ipu(preds, target_nan, average="weighted", num_classes=4, beta=0.5)
        self.assertFalse(score_true.isnan(), "Weighted FBETA_score multiclass with target_nan is NaN")
        self.assertFalse(
            score_ipu.isnan(), "Weighted FBETA_score multiclass IPU score with target_nan is NaN"
        )
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Regular FBETA_score multiclass with NaN is different",
        )

    def test_f1_score(self):
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

        # Micro f1_score binary
        score_true = f1_score(preds[:, 0], target.to(int) > 0, average="micro")
        score_ipu = f1_score_ipu(preds[:, 0], target > 0, average="micro")
        self.assertFalse(score_true.isnan(), "Micro F1_score binary is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro F1_score binary is different"
        )

        # Micro f1_score binary with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = f1_score(preds[:, 0][not_nan], target_nan_bin[not_nan].to(int), average="micro")
        score_ipu = f1_score_ipu(preds[:, 0], target_nan_bin, average="micro")
        self.assertFalse(score_true.isnan(), "Micro F1_score binary with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro F1_score binary IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro F1_score binary with NaN is different"
        )

        # Micro f1_score
        score_true = f1_score(preds, target.to(int), average="micro")
        score_ipu = f1_score_ipu(preds, target, average="micro")
        self.assertFalse(score_true.isnan(), "Micro F1_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro F1_score is different"
        )

        # Micro f1_score with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = f1_score(preds[not_nan], target[not_nan].to(int), average="micro")
        score_ipu = f1_score_ipu(preds, target_nan, average="micro")
        self.assertFalse(score_true.isnan(), "Micro F1_score with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Micro F1_score IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Micro F1_score with NaN is different"
        )

        # Macro f1_score multiclass
        score_true = f1_score(preds, target.to(int), average="macro", num_classes=4)
        score_ipu = f1_score_ipu(preds, target, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro F1_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Macro F1_score multiclass is different"
        )

        # Macro f1_score multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = f1_score(preds[not_nan], target[not_nan].to(int), average="macro", num_classes=4)
        score_ipu = f1_score_ipu(preds, target_nan, average="macro", num_classes=4)
        self.assertFalse(score_true.isnan(), "Macro F1_score multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Macro F1_score multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Macro F1_score multiclass with NaN is different",
        )

        # Weighted f1_scoremulticlass
        score_true = f1_score(preds, target.to(int), average="weighted", num_classes=4)
        score_ipu = f1_score_ipu(preds, target, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted F1_score is NaN")
        self.assertAlmostEqual(
            score_true.item(), score_ipu.item(), places=6, msg="Weighted F1_score is different"
        )

        # Weighted f1_score multiclass with NaNs in target
        not_nan = ~target_nan.isnan()
        score_true = f1_score(preds[not_nan], target[not_nan].to(int), average="weighted", num_classes=4)
        score_ipu = f1_score_ipu(preds, target_nan, average="weighted", num_classes=4)
        self.assertFalse(score_true.isnan(), "Weighted F1_score multiclass with target_nan is NaN")
        self.assertFalse(score_ipu.isnan(), "Weighted F1_score multiclass IPU score with target_nan is NaN")
        self.assertAlmostEqual(
            score_true.item(),
            score_ipu.item(),
            places=6,
            msg="Regular F1_score multiclass with NaN is different",
        )

    def test_mse(self):
        preds = deepcopy(self.preds)
        target = deepcopy(self.target)
        target_nan = deepcopy(self.target_nan)
        squared = True

        # Regular loss
        loss_true = mean_squared_error(preds, target, squared)
        loss_ipu = mean_squared_error_ipu(preds=preds, target=target, squared=squared)
        self.assertFalse(loss_true.isnan(), "Regular Mean Squared Error is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular Mean Squared Error is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = mean_squared_error(preds[not_nan], target[not_nan], squared)
        loss_ipu = mean_squared_error_ipu(preds=preds, target=target_nan, squared=squared)
        self.assertFalse(loss_true.isnan(), "Regular Mean Squared Error with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular Mean Squared Error IPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(),
            loss_ipu.item(),
            places=6,
            msg="Regular Mean Squared Error with NaN is different",
        )

        squared = False

        # Regular loss
        loss_true = mean_squared_error(preds, target, squared)
        loss_ipu = mean_squared_error_ipu(preds=preds, target=target, squared=squared)
        self.assertFalse(loss_true.isnan(), "Regular Mean Squared Error is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular Mean Squared Error is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = mean_squared_error(preds[not_nan], target[not_nan], squared)
        loss_ipu = mean_squared_error_ipu(preds=preds, target=target_nan, squared=squared)
        self.assertFalse(loss_true.isnan(), "Regular Mean Squared Error with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular Mean Squared Error IPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(),
            loss_ipu.item(),
            places=6,
            msg="Regular Mean Squared Error with NaN is different",
        )

    def test_mae(self):
        preds = deepcopy(self.preds)
        target = deepcopy(self.target)
        target_nan = deepcopy(self.target_nan)

        # Regular loss
        loss_true = mean_absolute_error(preds, target)
        loss_ipu = mean_absolute_error_ipu(preds=preds, target=target)
        self.assertFalse(loss_true.isnan(), "Regular Mean Absolute Error is NaN")
        self.assertAlmostEqual(
            loss_true.item(), loss_ipu.item(), places=6, msg="Regular Mean Absolute Error is different"
        )

        # Regular loss with NaNs in target
        not_nan = ~target_nan.isnan()
        loss_true = mean_absolute_error(preds[not_nan], target[not_nan])
        loss_ipu = mean_absolute_error_ipu(preds=preds, target=target_nan)
        self.assertFalse(loss_true.isnan(), "Regular Mean Absolute Error with target_nan is NaN")
        self.assertFalse(loss_ipu.isnan(), "Regular Mean Absolute Error IPU with target_nan is NaN")
        self.assertAlmostEqual(
            loss_true.item(),
            loss_ipu.item(),
            places=6,
            msg="Regular Mean Absolute Error with NaN is different",
        )


if __name__ == "__main__":
    ut.main()
