"""
Unit tests for the metrics and wrappers of goli/trainer/metrics/...
"""

import torch
import unittest as ut
from torch.nn import functional as F

from goli.trainer.losses import HybridCELoss


class test_Losses(ut.TestCase):
    input = torch.Tensor([[0.1, 0.1, 0.3, 0.5, 0.0], [0.1, 0.0, 0.7, 0.2, 0.0]])
    target = torch.Tensor([[0, 0, 0, 1, 0], [1, 0, 0, 0, 0]])
    brackets = torch.Tensor([0, 1, 2, 3, 4])
    regression_input = torch.Tensor([2.2, 2.0])  # inner product of input and brackets
    regression_target = torch.Tensor([3, 0])  # argmax over target

    def test_pure_ce_loss(self):
        loss = HybridCELoss(n_brackets=len(self.brackets), alpha=1.0, reduction="none")

        assert torch.equal(
            loss(self.input, self.target),
            F.cross_entropy(self.input, self.target, reduction="none"),
        )
        assert loss(self.input, self.target).shape == (2,)

    def test_pure_mae_loss(self):
        loss = HybridCELoss(
            n_brackets=len(self.brackets),
            alpha=0.0,
            regression_loss="mae",
            reduction="none",
        )

        assert torch.equal(
            loss(self.input, self.target),
            F.l1_loss(self.regression_input, self.regression_target, reduction="none"),
        )
        assert loss(self.input, self.target).shape == (2,)

    def test_pure_mse_loss(self):
        loss = HybridCELoss(
            n_brackets=len(self.brackets),
            alpha=0.0,
            regression_loss="mse",
            reduction="none",
        )

        assert torch.equal(
            loss(self.input, self.target),
            F.mse_loss(self.regression_input, self.regression_target, reduction="none"),
        )
        assert loss(self.input, self.target).shape == (2,)

    def test_hybrid_loss(self):
        loss = HybridCELoss(
            n_brackets=len(self.brackets), alpha=0.5, regression_loss="mse"
        )

        ce_loss = F.cross_entropy(self.input, self.target)
        mse_loss = F.mse_loss(self.regression_input, self.regression_target)

        assert torch.equal(
            loss(self.input, self.target), 0.5 * ce_loss + 0.5 * mse_loss
        )
        assert loss(self.input, self.target).shape == torch.Size([])


if __name__ == "__main__":
    ut.main()
