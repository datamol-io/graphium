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

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Parameters:
            input: (batch_size x n_classes) tensor of logits predicted for each bracket.
            target: (batch_size) or (batch_size, 1) tensor of target brackets in {0, 1, ..., self.n_brackets}.
        """

        target = target.flatten()
        # regression loss needs normalized logits to probability as input to do inner product with self.brackets
        # we apply softmax on the raw logits first
        softmax_input = self.softmax(input)
        # [batch_size, n_classes] * [n_classes] ([0, 1, 2...n_brakets-1]) -> [batch_size]
        regression_input = torch.inner(softmax_input, self.brackets.to(input.device))
        regression_loss = self.regression_loss(regression_input, target.float(), reduction=self.reduction)

        # cross_entropy loss needs raw logits as input
        ce_loss = F.cross_entropy(input, target.long(), weight=self.weight, reduction=self.reduction)

        return self.alpha * ce_loss + (1 - self.alpha) * regression_loss
