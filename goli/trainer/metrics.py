import torch
from torch.nn import functional as F
import torch.nn as nn
import operator as op

from pytorch_lightning.metrics.utils import reduce
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics import (
    Metric,
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1,
    FBeta,
    PrecisionRecallCurve,
    Precision,
    Recall,
    ROC,
    MeanAbsoluteError,
    MeanSquaredError,
)

EPS = 1e-5


class Thresholder(nn.Module):
    def __init__(self, threshold, operator="greater", th_on_pred=True, th_on_target=False):
        super().__init__()

        # Basic params
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_pred = th_on_pred

        # Operator can either be a string, or a callable
        if isinstance(operator, str):
            op_name = operator.lower()
            if op_name in ["greater", "gt"]:
                operator = op.gt
            elif op_name in ["lower", "lt"]:
                operator = op.lt
            else:
                raise ValueError(f"operator `{op_name}` not supported")
        elif callable(operator):
            pass
        else:
            raise TypeError(f"operator must be either `str` or `callable`, provided: `{type(operator)}`")

        self.operator = operator

    def forward(self, preds, target):
        # Apply the threshold on the predictions
        if self.th_on_pred:
            preds = self.operator(preds, self.threshold)

        # Apply the threshold on the targets
        if self.th_on_target:
            target = self.operator(target, self.threshold)

        return preds, target


class MetricWithThreshold(nn.Module):
    def __init__(self, metric, thresholder):
        super().__init__()
        self.metric = metric
        self.thresholder = thresholder

    def forward(self, preds, target):
        preds, target = self.thresholder.forward(preds, target)
        metric_val = self.metric.forward(preds, target)

        return metric_val


class MetricFunctionToClass(nn.Module):
    def __init__(self, function, name=None, **kwargs):
        super().__init__()
        self.name = name if name is not None else function.__name__
        self.function = function
        self.kwargs = kwargs

    def forward(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ):

        return self.function(preds=preds, target=target, **self.kwargs)


def pearsonr(preds: torch.Tensor, target: torch.Tensor, reduction: str = "elementwise_mean") -> torch.Tensor:
    """
    Computes the pearsonr correlation.

    Arguments
    ------------
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns
    -------------
        Tensor with the pearsonr

    Example:
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 2])
        >>> pearsonr(x, y)
        tensor(0.9439)
    """

    shifted_x = preds - torch.mean(preds, dim=0)
    shifted_y = target - torch.mean(target, dim=0)
    sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
    sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

    pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + EPS)
    pearson = torch.clamp(pearson, min=-1, max=1)
    pearson = reduce(pearson, reduction=reduction)
    return pearson


def spearmanr(preds: torch.Tensor, target: torch.Tensor, reduction: str = "elementwise_mean") -> torch.Tensor:
    """
    Computes the spearmanr correlation.

    Arguments
    ------------
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns
    -------------
        Tensor with the spearmanr

    Example:
        >>> x = torch.tensor([0., 1, 2, 3])
        >>> y = torch.tensor([0., 1, 2, 1.5])
        >>> spearmanr(x, y)
        tensor(0.8)
    """

    pred_rank = torch.argsort(preds, dim=0).float()
    target_rank = torch.argsort(target, dim=0).float()
    spearman = pearsonr(pred_rank, target_rank, reduction=reduction)
    return spearman


METRICS_DICT = {
    "accuracy": Accuracy,
    "averageprecision": AveragePrecision,
    "auroc": auroc,
    "confusionmatrix": ConfusionMatrix,
    "f1": F1,
    "fbeta": FBeta,
    "precisionrecallcurve": PrecisionRecallCurve,
    "precision": Precision,
    "recall": Recall,
    "roc": ROC,
    "mae": MeanAbsoluteError,
    "mse": MeanSquaredError,
    "pearsonr": pearsonr,
    "spearmanr": spearmanr,
}


if __name__ == "__main__":
    preds = torch.tensor([0.0, 1, 2, 3])
    target = torch.tensor([0.0, 1, 2, 1.5])
    print(spearmanr(preds, target))

    sp = MetricFunctionToClass(spearmanr)
    print(sp(preds, target))
