from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss


class HybridCELoss(_WeightedLoss):
    def __init__(
        self,
        n_brackets,
        regression_loss: str = "mse",
        alpha: float = 0.5,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        """
        A hybrid between the regression loss (either MAE or MSE) and the cross entropy loss. Intended
        to be used with noisy regression datasets, for which the targets are assigned to binary brackets,
        and the task is transformed into a multi-class classification.

        Note that it assumes that the brackets are consecutive integers starting at 0 up to n_brackets,
        which has an impact on the scale of the regression loss component.

        Parameters:
            n_brackets: the number of brackets that will be used to group the regression targets.
                Expected to have the same size as the number of classes in the transformed regression task.
            regression_loss: type of regression loss, either 'mse' or 'mae'.
            alpha: weight assigned to the CE loss component. Must be a value in [0, 1] range.
            weight: a manual rescaling weight given to each class in the CE loss component.
                If given, has to be a Tensor of the same size as the number of classes.
            reduction: specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                'none': no reduction will be applied, 'mean': the sum of the output will be divided
                by the number of elements in the output, 'sum': the output will be summed.
        """
        super().__init__(weight=weight, reduction=reduction)

        if regression_loss == "mae":
            self.regression_loss = F.l1_loss
        elif regression_loss == "mse":
            self.regression_loss = F.mse_loss
        else:
            raise ValueError(
                f"Expected regression_loss to be in {{'mae', 'mse'}}, received {regression_loss}."
            )

        if alpha < 0 or alpha > 1:
            raise ValueError(f"Expected alpha to be in the [0, 1] range, received {alpha}.")

        self.brackets = Tensor(range(n_brackets))
        self.alpha = alpha
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input: Tensor, target: Tensor, nan_targets: Tensor = None) -> Tensor:
        """
        Parameters:
            input: (batch_size x n_classes) tensor of logits predicted for each bracket.
            target: (batch_size) or (batch_size, 1) tensor of target brackets in {0, 1, ..., self.n_brackets}.
        """

        target = target.flatten()

        # set input and target with nans to 0s for regression loss
        if nan_targets is not None:
            input = torch.masked_fill(input, nan_targets.unsqueeze(1), 0)
            target = torch.masked_fill(target, nan_targets, 0)
        # regression loss needs normalized logits to probability as input to do inner product with self.brackets
        # we apply softmax on the raw logits first
        softmax_input = self.softmax(input)
        # the softmax of a tensor of 0s would not be 0s anymore, so need to apply nan_targets here again to filter out
        if nan_targets is not None:
            softmax_input = torch.masked_fill(softmax_input, nan_targets.unsqueeze(1), 0.0)
        # [batch_size, n_classes] * [n_classes] ([0, 1, 2...n_brakets-1]) -> [batch_size]
        regression_input = torch.inner(softmax_input.to(self.brackets.dtype), self.brackets.to(input.device))
        regression_loss = self.regression_loss(regression_input, target.float(), reduction=self.reduction)
        # regression_loss needs some scaling by total_targets/num_real_targets
        if nan_targets is not None:
            num_real_targets = (~nan_targets).sum()
            factor1 = torch.where(num_real_targets > 0, 1, 0)
            factor2 = torch.where(num_real_targets > 0, 0, 1)
            regression_loss = factor1 * regression_loss * nan_targets.numel() / (num_real_targets + factor2)

            # set input and target with nans to -1000s for ce loss
            input = torch.masked_fill(input, nan_targets.unsqueeze(1), -1000)
            target = torch.masked_fill(target, nan_targets, -1000)
        # cross_entropy loss needs raw logits as input
        # ce_loss does not need scaling as it already ignores -1000 masked nan values
        ce_loss = F.cross_entropy(
            input, target.long(), weight=self.weight, ignore_index=-1000, reduction=self.reduction
        )

        return self.alpha * ce_loss + (1 - self.alpha) * regression_loss
