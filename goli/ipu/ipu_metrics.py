import torch
from torch import Tensor
from torch.nn import BCELoss, MSELoss, L1Loss
from torchmetrics.functional import auroc, average_precision, precision, accuracy, recall
from torch._C import _infer_size

from typing import Optional, Sequence


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

    target = target.clone()
    preds = preds.clone()

    nans = torch.isnan(target)
    target[nans] = 1
    preds[nans] = 0

    # Compute the loss, and rescale by the number of nan elements
    score = precision (
        preds = preds,
        target = target.to(int),
        average = average,
        mdmc_average = mdmc_average,
        ignore_index = ignore_index,
        num_classes = num_classes,
        threshold = threshold,
        top_k = top_k,
        multiclass = multiclass)

    return score


class NaNTensor(Tensor):
    def set_nans(self, nans):
        self._nans = nans
        return self

    @property
    def get_nans(self):
        return self._nans
        # if self.is_floating_point():
        #     return self.isnan()
        # else:
        #     return Tensor.__eq__(self, torch.iinfo(self.dtype).min)

    def sum(self, *args, **kwargs):
        tensor = self
        tensor[tensor.get_nans] = float("nan")
        return tensor.nansum(*args, **kwargs).to(self.dtype)
    def min(self, *args, **kwargs):
        tensor = self
        tensor = tensor[~self.get_nans]
        return super(NaNTensor, tensor).min(*args, **kwargs)
    def max(self, *args, **kwargs):
        tensor = self
        tensor = tensor[~self.get_nans]
        return super(NaNTensor, tensor).max(*args, **kwargs)
    def __eq__(self, other) -> Tensor:
        if isinstance(self, NaNTensor) and isinstance(other, NaNTensor):
            is_eq = super().__eq__(other).to(torch.float32)
            is_eq[self.get_nans | other.get_nans] = float("nan")
            return is_eq
        else:
            return super().__eq__(other) & ~self.get_nans

    def __ne__(self, other) -> Tensor:
        if isinstance(self, NaNTensor) and isinstance(other, NaNTensor):
            is_ne = super().__ne__(other).to(torch.float32)
            is_ne[self.get_nans | other.get_nans] = float("nan")
            return is_ne
        else:
            return super().__ne__(other) | self.get_nans

    def to(self, *args, **kwargs):
        nans = self.get_nans
        return super().to(*args, **kwargs).set_nans(nans)

    def clone(self, *args, **kwargs) -> Tensor:
        nans = self.get_nans
        return super().clone(*args, **kwargs).set_nans(nans)

    def squeeze(self, *args, **kwargs):
        nans = self.get_nans.squeeze(*args, **kwargs)
        return super().squeeze(*args, **kwargs).set_nans(nans)

    def unsqueeze(self, *args, **kwargs):
        nans = self.get_nans.unsqueeze(*args, **kwargs)
        return super().unsqueeze(*args, **kwargs).set_nans(nans)


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
    ignore_index: Optional[int] = None
    ):
    """
    A modified version of the `torchmetrics.functional.precision` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.

    1/N * sum_N(preds==target)
    """

    nans = torch.isnan(target)
    target = NaNTensor(target.clone()).set_nans(nans).to(int)
    preds = NaNTensor(preds.clone()).set_nans(nans).to(int)

    # target[nans] = 0
    # preds[nans] = 0

    # if num_classes is not None:
    #     num_classes += 1

    # Compute the loss, and rescale by the number of nan elements
    score = accuracy (
        preds = preds,
        target = target,
        average = average,
        mdmc_average = mdmc_average,
        threshold = threshold,
        top_k = top_k,
        subset_accuracy = subset_accuracy,
        num_classes = num_classes,
        multiclass = multiclass,
        ignore_index = ignore_index,
        )

    return score

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

    target = target.clone()
    preds = preds.clone()

    nans = torch.isnan(target)
    target[nans] = 0
    preds[nans] = 1

    # Compute the loss, and rescale by the number of nan elements
    score = recall (
        preds = preds,
        target = target.to(int),
        average = average,
        mdmc_average = mdmc_average,
        ignore_index = ignore_index,
        num_classes = num_classes,
        threshold = threshold,
        top_k = top_k,
        multiclass = multiclass)

    return score
