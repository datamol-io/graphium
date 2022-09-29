import torch
from torch import Tensor
from torchmetrics.functional import auroc, average_precision, pearson_corrcoef, r2_score # Remove imports

from typing import Optional, Sequence

from torchmetrics.utilities.checks import _input_squeeze
from torchmetrics.functional.classification.accuracy import _mode, _check_subset_validity, _accuracy_compute, _accuracy_update # Remove imports
from torchmetrics.functional.classification.precision_recall import _precision_compute, _recall_compute
from torchmetrics.functional.classification.f_beta import f1_score, _fbeta_compute
from torchmetrics.utilities.checks import _input_squeeze # Remove imports

from goli.utils.tensor import nan_mean

# Move losses into ipu_loss file and add MSE and MAE implementations, do as MSE LOSS, add same tests as for test_mse, jsut replace with MSE Loss (re-implement the metric)

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

def r2_score_ipu(preds, target, *args, **kwargs):
    preds = NaNTensor(preds)
    target = NaNTensor(target)
    preds[target.get_nans] = float("nan")
    score = r2_score(preds, target, *args, **kwargs)
    return Tensor(score)

def f1_score_ipu(
    preds: Tensor,
    target: Tensor,
    beta: float = 1.0,
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
        preds = preds,
        target=target,
        average = average,
        mdmc_average = mdmc_average,
        ignore_index = ignore_index,
        num_classes = num_classes,
        threshold = threshold,
        top_k = top_k,
        multiclass = multiclass
        )

    return _fbeta_compute(tp, fp, tn, fn, 1.0, ignore_index, average, mdmc_average)