"""
Unit tests for the metrics and wrappers of graphium/trainer/metrics/...
"""

import torch
import unittest as ut
from torch.nn import functional as F

from graphium.trainer.losses import HybridCELoss
from graphium.trainer.predictor_options import EvalOptions


def _parse(loss_fun):
    eval_options = EvalOptions(loss_fun=loss_fun, metrics_on_progress_bar=None)
    return eval_options.parse_loss_fun(loss_fun)


class test_HybridCELoss(ut.TestCase):
    input = torch.Tensor([[0.1, 0.1, 0.3, 0.5, 0.0], [0.1, 0.0, 0.7, 0.2, 0.0]])
    target = torch.Tensor([3, 0]).long()
    brackets = torch.Tensor([0, 1, 2, 3, 4])
    regression_input = torch.Tensor([2.0537, 2.0017])  # inner product of input and brackets
    regression_target = torch.Tensor([3, 0]).float()

    def test_pure_ce_loss(self):
        loss = HybridCELoss(n_brackets=len(self.brackets), alpha=1.0, reduction="none")
        assert torch.allclose(
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
        assert torch.allclose(
            loss(self.input, self.target),
            F.l1_loss(self.regression_input, self.regression_target, reduction="none"),
            rtol=1e-04,
            atol=1e-07,
        )
        assert loss(self.input, self.target).shape == (2,)

    def test_pure_mse_loss(self):
        loss = HybridCELoss(
            n_brackets=len(self.brackets),
            alpha=0.0,
            regression_loss="mse",
            reduction="none",
        )

        assert torch.allclose(
            loss(self.input, self.target),
            F.mse_loss(self.regression_input, self.regression_target, reduction="none"),
            rtol=1e-04,
            atol=1e-07,
        )
        assert loss(self.input, self.target).shape == (2,)

    def test_hybrid_loss(self):
        loss = HybridCELoss(n_brackets=len(self.brackets), alpha=0.5, regression_loss="mse")

        ce_loss = F.cross_entropy(self.input, self.target)
        mse_loss = F.mse_loss(self.regression_input, self.regression_target)

        assert torch.allclose(
            loss(self.input, self.target), 0.5 * ce_loss + 0.5 * mse_loss, rtol=1e-04, atol=1e-07
        )
        assert loss(self.input, self.target).shape == torch.Size([])

    def test_loss_parser(self):
        # HybridCE cannot be parsed from a string because it requires specifying n_brackets
        loss_fun = "hybrid_ce"
        self.assertRaises(TypeError, _parse, loss_fun=loss_fun)

        # HybridCE requires n_brackets to be specified
        loss_fun = {"name": "hybrid_ce"}
        self.assertRaises(TypeError, _parse, loss_fun=loss_fun)

        loss_fun = {"name": "hybrid_ce", "n_brackets": 3}
        parsed = _parse(loss_fun)

        assert isinstance(parsed, HybridCELoss)
        assert len(parsed.brackets) == 3
        assert parsed.regression_loss == F.mse_loss

        loss_fun = {"name": "hybrid_ce", "n_brackets": 5, "regression_loss": "mae"}
        parsed = _parse(loss_fun)

        assert isinstance(parsed, HybridCELoss)
        assert len(parsed.brackets) == 5
        assert parsed.regression_loss == F.l1_loss


class test_BCELoss(ut.TestCase):
    def test_loss_parser(self):
        loss_fun = "bce"
        parsed = _parse(loss_fun)

        assert isinstance(parsed, torch.nn.BCELoss)

        loss_fun = {"name": "bce"}
        parsed = _parse(loss_fun)

        assert isinstance(parsed, torch.nn.BCELoss)
        assert parsed.reduction != "sum"

        loss_fun = {"name": "bce", "reduction": "sum"}
        parsed = _parse(loss_fun)

        assert isinstance(parsed, torch.nn.BCELoss)
        assert parsed.reduction == "sum"


if __name__ == "__main__":
    ut.main()
