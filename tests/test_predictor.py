"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


"""
Unit tests for the file graphium/trainer/predictor.py
"""

import torch
from torch.nn import BCELoss, MSELoss
import unittest as ut

from graphium.trainer.predictor_options import EvalOptions


class test_Predictor(ut.TestCase):
    def test_parse_loss_fun(self):
        losses = ["bce", "mse", "mae", "l1", BCELoss(), MSELoss()]
        preds = torch.rand(10, 5)
        target = (torch.rand(10, 5) > 0.5).to(preds.dtype)
        for this_loss in losses:
            loss_fun = EvalOptions.parse_loss_fun(this_loss)
            loss = loss_fun(preds, target)


if __name__ == "__main__":
    ut.main()
