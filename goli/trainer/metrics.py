from typing import Union, Callable, Optional, Dict, Any

from copy import deepcopy
import torch
from torch.nn import functional as F
import torch.nn as nn
import operator as op

from pytorch_lightning.metrics.utils import reduce
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import (
    accuracy,
    average_precision,
    confusion_matrix,
    f1,
    fbeta,
    precision_recall_curve,
    precision,
    recall,
    auroc,
    multiclass_auroc,
    mean_absolute_error,
    mean_squared_error,
)

EPS = 1e-5


class Thresholder:
    def __init__(
        self,
        threshold: float,
        operator: str = "greater",
        th_on_preds: bool = True,
        th_on_target: bool = False,
        target_to_int: bool = False,
    ):

        # Basic params
        self.threshold = threshold
        self.th_on_target = th_on_target
        self.th_on_preds = th_on_preds
        self.target_to_int = target_to_int

        # Operator can either be a string, or a callable
        if isinstance(operator, str):
            op_name = operator.lower()
            if op_name in ["greater", "gt"]:
                op_str = ">"
                operator = op.gt
            elif op_name in ["lower", "lt"]:
                op_str = "<"
                operator = op.lt
            else:
                raise ValueError(f"operator `{op_name}` not supported")
        elif callable(operator):
            op_str = operator.__name__
        elif operator is None:
            pass
        else:
            raise TypeError(f"operator must be either `str` or `callable`, provided: `{type(operator)}`")

        self.operator = operator
        self.op_str = op_str

    def compute(self, preds: torch.Tensor, target: torch.Tensor):
        # Apply the threshold on the predictions
        if self.th_on_preds:
            preds = self.operator(preds, self.threshold)

        # Apply the threshold on the targets
        if self.th_on_target:
            target = self.operator(target, self.threshold)

        if self.target_to_int:
            target = target.to(int)

        return preds, target

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """

        return f"{self.op_str}{self.threshold}"


def pearsonr(preds: torch.Tensor, target: torch.Tensor, reduction: str = "elementwise_mean") -> torch.Tensor:
    r"""
    Computes the pearsonr correlation.

    Parameters:
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns:
        Tensor with the pearsonr

    !!! Example
        ``` python linenums="1"
        x = torch.tensor([0., 1, 2, 3])
        y = torch.tensor([0., 1, 2, 2])
        pearsonr(x, y)
        >>> tensor(0.9439)
        ```
    """

    preds, target = preds.to(torch.float32), target.to(torch.float32)

    shifted_x = preds - torch.mean(preds, dim=0)
    shifted_y = target - torch.mean(target, dim=0)
    sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
    sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

    pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + EPS)
    pearson = torch.clamp(pearson, min=-1, max=1)
    pearson = reduce(pearson, reduction=reduction)
    return pearson


def _get_rank(values):

    arange = torch.arange(values.shape[0], dtype=values.dtype, device=values.device)

    val_sorter = torch.argsort(values, dim=0)
    val_rank = torch.empty_like(values)
    if values.ndim == 1:
        val_rank[val_sorter] = arange
    elif values.ndim == 2:
        for ii in range(val_rank.shape[1]):
            val_rank[val_sorter[:, ii], ii] = arange
    else:
        raise ValueError(f"Only supports tensors of dimensions 1 and 2, provided dim=`{preds.ndim}`")

    return val_rank


def spearmanr(preds: torch.Tensor, target: torch.Tensor, reduction: str = "elementwise_mean") -> torch.Tensor:
    r"""
    Computes the spearmanr correlation.

    Parameters:
        preds: estimated labels
        target: ground truth labels
        reduction: a method to reduce metric score over labels.
            - ``'elementwise_mean'``: takes the mean (default)
            - ``'sum'``: takes the sum
            - ``'none'``: no reduction will be applied

    Returns:
        Tensor with the spearmanr

    !!! Example
        x = torch.tensor([0., 1, 2, 3])
        y = torch.tensor([0., 1, 2, 1.5])
        spearmanr(x, y)
        tensor(0.8)
    """

    spearman = pearsonr(_get_rank(preds), _get_rank(target), reduction=reduction)
    return spearman


METRICS_CLASSIFICATION = {
    "accuracy": accuracy,
    "averageprecision": average_precision,
    "auroc": auroc,
    "confusionmatrix": confusion_matrix,
    "f1": f1,
    "fbeta": fbeta,
    "precisionrecallcurve": precision_recall_curve,
    "precision": precision,
    "recall": recall,
    "multiclass_auroc": multiclass_auroc,
}

METRICS_REGRESSION = {
    "mae": mean_absolute_error,
    "mse": mean_squared_error,
    "pearsonr": pearsonr,
    "spearmanr": spearmanr,
}

METRICS_DICT = deepcopy(METRICS_CLASSIFICATION)
METRICS_DICT.update(METRICS_REGRESSION)


class MetricWrapper:
    r"""
    Allows to initialize a metric from a name or Callable, and initialize the
    `Thresholder` in case the metric requires a threshold.
    """

    def __init__(
        self, metric: Union[str, Callable], threshold_kwargs: Optional[Dict[str, Any]] = None, **kwargs
    ):
        r"""
        Parameters
            metric:
                The metric to use. See `METRICS_DICT`

            threshold_kwargs:
                If `None`, no threshold is applied.
                Otherwise, we use the class `Thresholder` is initialized with the
                provided argument, and called before the `compute`

            kwargs:
                Other arguments to call with the metric
        """

        self.metric = METRICS_DICT[metric] if isinstance(metric, str) else metric

        self.thresholder = None
        if threshold_kwargs is not None:
            self.thresholder = Thresholder(**threshold_kwargs)

        self.kwargs = kwargs

    def compute(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric, and apply the thresholder if provided
        """
        if self.thresholder is not None:
            preds, target = self.thresholder(preds, target)
        metric_val = self.metric(preds, target, **self.kwargs)
        return metric_val

    def __call__(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the metric with the method `self.compute`
        """
        return self.compute(preds, target)

    def __repr__(self):
        r"""
        Control how the class is printed
        """
        full_str = f"{self.metric.__name__}"
        if self.thresholder is not None:
            full_str += f"({self.thresholder})"

        return full_str
