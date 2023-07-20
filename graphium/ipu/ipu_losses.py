import torch
from torch import Tensor
from torch.nn import BCELoss, BCEWithLogitsLoss, MSELoss, L1Loss
from torch._C import _infer_size
from loguru import logger
from graphium.trainer.losses import HybridCELoss


class BCEWithLogitsLossIPU(BCEWithLogitsLoss):
    """
    A modified version of the `torch.nn.BCEWithLogitsLoss` that can ignore NaNs
    by giving them a weight of `0`. This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        prev_weight = None

        target = target.clone().to(input.dtype)
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

        num_real_targets = (~nan_targets).sum()
        factor1 = torch.where(num_real_targets > 0, 1, 0)
        factor2 = torch.where(num_real_targets > 0, 0, 1)
        loss = factor1 * loss * nan_targets.numel() / (num_real_targets + factor2)

        # Reset the self.weight to its original value
        self.weight = prev_weight

        return loss


class BCELossIPU(BCELoss):
    """
    A modified version of the `torch.nn.BCELoss` that can ignore NaNs
    by giving them a weight of `0`. This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        prev_weight = None

        target = target.clone().to(input.dtype)
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

        num_real_targets = (~nan_targets).sum()
        factor1 = torch.where(num_real_targets > 0, 1, 0)
        factor2 = torch.where(num_real_targets > 0, 0, 1)
        loss = factor1 * loss * nan_targets.numel() / (num_real_targets + factor2)

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
        target = target.clone().to(input.dtype)
        input = input.clone()

        # Replace the nan-targets in the input/target tensors by 0
        nan_targets = target.isnan()
        input[nan_targets] = 0.0
        target[nan_targets] = 0.0

        # Compute the loss, and rescale by the number of nan elements
        loss = super().forward(input, target)

        num_real_targets = (~nan_targets).sum()
        factor1 = torch.where(num_real_targets > 0, 1, 0)
        factor2 = torch.where(num_real_targets > 0, 0, 1)
        loss = factor1 * loss * nan_targets.numel() / (num_real_targets + factor2)

        return loss


class L1LossIPU(L1Loss):
    """
    A modified version of the `torch.nn.L1Loss` that can ignore NaNs
    by giving them the same value for both `input` and `target`.
    This allows it to work with compilation
    and IPUs since it doesn't modify the tensor's shape.
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.clone().to(input.dtype)
        input = input.clone()

        # Replace the nan-targets in the input/target tensors by 0
        nan_targets = target.isnan()
        input[nan_targets] = 0.0
        target[nan_targets] = 0.0

        # Compute the loss, and rescale by the number of nan elements
        loss = super().forward(input, target)
        num_real_targets = (~nan_targets).sum()
        factor1 = torch.where(num_real_targets > 0, 1, 0)
        factor2 = torch.where(num_real_targets > 0, 0, 1)
        loss = factor1 * loss * nan_targets.numel() / (num_real_targets + factor2)

        return loss


class HybridCELossIPU(HybridCELoss):
    def __init__(
        self,
        n_brackets,
        alpha: float = 0.5,
    ) -> None:
        """
        Parameters:
            n_brackets: the number of brackets that will be used to group the regression targets.
                Expected to have the same size as the number of classes in the transformed regression task.
        """
        super().__init__(n_brackets=n_brackets, alpha=alpha)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Parameters:
            input: (batch_size x n_classes) tensor of logits predicted for each bracket.
            target: (batch_size) or (batch_size, 1) tensor of target brackets in {0, 1, ..., self.n_brackets}.
        """

        target = target.clone().to(input.dtype)
        input = input.clone()

        # Replace the nan-targets in the input/target tensors by 0
        nan_targets = target.isnan()

        # Compute the loss, and rescale by the number of nan elements
        loss = super().forward(input, target, nan_targets)
        return loss
