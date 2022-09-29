import torch
from torch import Tensor
from torch.nn import BCELoss, MSELoss, L1Loss
from torchmetrics.functional import auroc, average_precision, precision, accuracy, recall, pearson_corrcoef, spearman_corrcoef, r2_score
from torch._C import _infer_size

from typing import Optional, Sequence

from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.functional.classification.accuracy import _mode, _check_subset_validity, _subset_accuracy_compute, _subset_accuracy_update, _accuracy_compute, _accuracy_update
from torchmetrics.functional.classification.precision_recall import _precision_compute, _recall_compute
from torchmetrics.utilities.checks import _check_classification_inputs, _input_format_classification, _input_squeeze
from torchmetrics.utilities.enums import AverageMethod, DataType, MDMCAverageMethod

from goli.utils.tensor import nan_mean

class BCELossIPU(BCELoss):
    """
    A modified version of the `torch.nn.BCELoss` that can ignore NaNs
    by giving them a weight of `0`. This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        prev_weight = None

        target = target.clone()
        weight = self.weight

        # Get the original weight matrix. If None, set all weights = 1
        if weight is not None:
            prev_weight = self.weight.clone()
            new_size = _infer_size(target.size(), weight.size())
            weight = weight.expand(new_size).clone()
        else:
            weight = torch.ones(target.shape, dtype=input.dtype, device=input.device)

        # Replace the nan-targets by 0 or 1. Take the value closest to the input.
        # Give a weight of 0 where there are nan-targets
        nan_targets = target.isnan()
        nan_targets_0 = (input < 0.5) & nan_targets
        nan_targets_1 = (input >= 0.5) & nan_targets
        target[nan_targets_0] = 0.0
        target[nan_targets_1] = 1.0
        weight[nan_targets] = 0.0

        # Compute the loss, and rescale by the number of nan elements
        self.weight = weight
        loss = super().forward(input, target)
        loss = loss * nan_targets.numel() / ((~nan_targets).sum())

        # Reset the self.weight to its original value
        self.weight = prev_weight
        return loss


class MSELossIPU(MSELoss):
    """
    A modified version of the `torch.nn.MSELoss` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        target = target.clone()
        input = input.clone()

        # Replace the nan-targets in the input/target tensors by 0
        nan_targets = target.isnan()
        input[nan_targets] = 0.0
        target[nan_targets] = 0.0

        # Compute the loss, and rescale by the number of nan elements
        loss = super().forward(input, target)
        loss = loss * nan_targets.numel() / ((~nan_targets).sum())

        return loss


class L1LossIPU(L1Loss):
    """
    A modified version of the `torch.nn.L1Loss` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        target = target.clone()
        input = input.clone()

        # Replace the nan-targets in the input/target tensors by 0
        nan_targets = target.isnan()
        input[nan_targets] = 0.0
        target[nan_targets] = 0.0

        # Compute the loss, and rescale by the number of nan elements
        loss = super().forward(input, target)
        loss = loss * nan_targets.numel() / ((~nan_targets).sum())

        return loss


def auroc_ipu(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    max_fpr: Optional[float] = None,
    sample_weights: Optional[Sequence] = None
    ):
    """
    A modified version of the `torchmetrics.functional.auroc` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    target = target.clone()
    preds = preds.clone()

    # Replace the nan-targets in the preds/target tensors by 0
    nan_targets = target.isnan()
    preds[nan_targets] = 0.0
    target[nan_targets] = 0.0

    # Get the original weight matrix. If None, set all weights = 1
    if sample_weights is None:
        sample_weights = torch.ones(target.shape[0], dtype=preds.dtype, device=preds.device)
    sample_weights[nan_targets] = 0.0

    # Compute the loss, and rescale by the number of nan elements
    score = auroc(
        preds = preds,
        target = target.to(int),
        num_classes = num_classes,
        pos_label = pos_label,
        average = average,
        max_fpr = max_fpr,
        sample_weights = sample_weights
    )

    return score

def average_precision_ipu(
    preds: Tensor,
    target: Tensor,
    num_classes: Optional[int] = None,
    pos_label: Optional[int] = None,
    average: Optional[str] = "macro",
    sample_weights: Optional[Sequence] = None,
    ):
    """
    A modified version of the `torchmetrics.functional.average_precision` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    target = target.clone()
    preds = preds.clone()

    # Replace the nan-targets in the preds/target tensors by 0
    nan_targets = target.isnan()
    preds[nan_targets] = 0.0
    target[nan_targets] = 0.0

    # Get the original weight matrix. If None, set all weights = 1
    if sample_weights is None:
        sample_weights = torch.ones(target.shape[0], dtype=preds.dtype, device=preds.device)
    sample_weights[nan_targets] = 0.0

    # Compute the loss, and rescale by the number of nan elements
    score = average_precision (
        preds = preds,
        target = target.to(int),
        num_classes = num_classes,
        pos_label = pos_label,
        average = average,
        sample_weights = sample_weights)

    return score

def precision_ipu(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ):
    """
    A modified version of the `torchmetrics.functional.precision` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        threshold=threshold,
        top_k=top_k,
        subset_accuracy=False,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
        )

    return _precision_compute(tp, fp, fn, average, mdmc_average)



def recall_ipu(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = None,
    ignore_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None
    ):
    """
    A modified version of the `torchmetrics.functional.precision` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        threshold=threshold,
        top_k=top_k,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
        )

    return _recall_compute(tp, fp, fn, average, mdmc_average)


def accuracy_ipu(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
    ):
    """
    A modified version of the `torchmetrics.functional.accuracy` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    (tp, fp, tn, fn), mode = get_confusion_matrix(
        preds=preds,
        target=target,
        average=average,
        mdmc_average=mdmc_average,
        threshold=threshold,
        top_k=top_k,
        subset_accuracy=subset_accuracy,
        num_classes=num_classes,
        multiclass=multiclass,
        ignore_index=ignore_index,
        )

    return _accuracy_compute(tp, fp, tn, fn, average, mdmc_average, mode)


def get_confusion_matrix(
    preds: Tensor,
    target: Tensor,
    average: Optional[str] = "micro",
    mdmc_average: Optional[str] = "global",
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    subset_accuracy: bool = False,
    num_classes: Optional[int] = None,
    multiclass: Optional[bool] = None,
    ignore_index: Optional[int] = None,
):
    allowed_average = ["micro", "macro", "weighted", "samples", "none", None]
    if average not in allowed_average:
        raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

    if average in ["macro", "weighted", "none", None] and (not num_classes or num_classes < 1):
        raise ValueError(f"When you set `average` as {average}, you have to provide the number of classes.")

    allowed_mdmc_average = [None, "samplewise", "global"]
    if mdmc_average not in allowed_mdmc_average:
        raise ValueError(f"The `mdmc_average` has to be one of {allowed_mdmc_average}, got {mdmc_average}.")

    if num_classes and ignore_index is not None and (not ignore_index < num_classes or num_classes == 1):
        raise ValueError(f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes")

    if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
        raise ValueError(f"The `top_k` should be an integer larger than 0, got {top_k}")

    #### ADDED ####
    # Put all the NaNs as the 0-class
    nans = torch.isnan(target)
    target[nans] = 0
    preds[nans] = 0
    if (preds.ndim > 1) and (preds.shape[1] > 1):
        preds[nans, 0] = 1
    target = target.to(int)
    #### END ADDED ####

    preds, target = _input_squeeze(preds, target)
    mode = _mode(preds, target, threshold, top_k, num_classes, multiclass, ignore_index)
    reduce = "macro" if average in ["weighted", "none", None] else average

    if subset_accuracy and _check_subset_validity(mode):
        # correct, total = _subset_accuracy_update(preds, target, threshold, top_k, ignore_index)
        # return _subset_accuracy_compute(correct, total)
        raise NotImplementedError("subset_accuracy not implemented")
    tp, fp, tn, fn = _accuracy_update(
        preds, target, reduce, mdmc_average, threshold, num_classes, top_k, multiclass, ignore_index, mode
    )

    #### ADDED ####
    num_nans = nans.sum(0)
    if tp.numel() > 1:
        tp[0] = tp[0] - num_nans
        tn[1:] = tn[1:] - num_nans
    else:
        tn = tn - num_nans
        if (preds.ndim > 1) and (preds.shape[1] > 1):
            tp = tp - num_nans
    #### END ADDED ####

    return (tp, fp, tn, fn), mode


class NaNTensor(Tensor):
    @property
    def get_nans(self):
        if self.is_floating_point():
            return self.isnan()
        elif self.is_signed():
            return self == torch.iinfo(self.dtype).min
        else:
            return torch.zeros(self.shape, device=self.device, dtype=bool)

    def sum(self, *args, **kwargs):
        tensor = self.to(float)
        tensor[self.get_nans] = float("nan")
        if self.is_floating_point():
            dtype = self.dtype
        else:
            dtype = torch.int64
        return tensor.nansum(*args, **kwargs).to(dtype)
    def mean(self, *args, **kwargs):
        tensor = self.to(float)
        tensor[self.get_nans] = float("nan")
        return nan_mean(tensor, *args, **kwargs).to(self.dtype)
    def numel(self):
        return super(NaNTensor, ~self.get_nans).sum()
    def min(self, *args, **kwargs):
        tensor = self
        tensor = tensor[~self.get_nans]
        return super(NaNTensor, tensor).min(*args, **kwargs)
    def max(self, *args, **kwargs):
        tensor = self
        tensor = tensor[~self.get_nans]
        return super(NaNTensor, tensor).max(*args, **kwargs)
    def argsort(self, dim=-1, descending=False):
        tensor = self
        if descending:
            tensor[tensor.get_nans] = float("-inf")
        else:
            tensor[tensor.get_nans] = float("inf")
        return super(NaNTensor, tensor).argsort(dim=dim, descending=descending)
    def size(self, dim):
        return (~self.get_nans).sum(dim=dim)

    def __lt__(self, other) -> Tensor:
        if other == 2:
            return super().__lt__(other).all()
        else:
            return super().__lt__(other)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.

        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.

        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.

        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        if func.__name__ == "sum":
            kwargs = {} if kwargs is None else kwargs
            return args[0].sum(*args[1:], **kwargs)
        else:
            return super().__torch_function__(func, types, args=args, kwargs=kwargs)

def pearson_ipu(preds, target):
    preds = NaNTensor(preds)
    target = NaNTensor(target)
    preds[target.get_nans] = float("nan")
    pearson = pearson_corrcoef(preds, target)
    return Tensor(pearson)


def spearman_ipu(preds, targets):
    raise NotImplementedError("SpearmanR cannot work due to indexing")
    # preds = NaNTensor(preds)
    # targets = NaNTensor(targets)
    # preds[targets.get_nans] = float("nan")
    # spearman = spearman_corrcoef(preds, targets)
    # return Tensor(spearman)

def r2_score_ipu(preds, target, *args, **kwargs):
    preds = NaNTensor(preds)
    target = NaNTensor(target)
    preds[target.get_nans] = float("nan")
    score = r2_score(preds, target, *args, **kwargs)
    return Tensor(score)

