"""
Unit tests for the file goli/trainer/predictor.py
"""

import torch
from torch.nn import BCELoss, MSELoss
import unittest as ut

from goli.trainer.predictor import PredictorModule


class test_Predictor(ut.TestCase):
    def test_parse_loss_fun(self):
        losses = ["bce", "mse", "mae", "l1", BCELoss(), MSELoss()]
        preds = torch.rand(10, 5)
        target = (torch.rand(10, 5) > 0.5).to(preds.dtype)
        for this_loss in losses:
            loss_fun = PredictorModule.parse_loss_fun(this_loss)
            loss = loss_fun(preds, target)


if __name__ == "__main__":
    ut.main()
