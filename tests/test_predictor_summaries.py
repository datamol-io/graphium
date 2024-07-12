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
from torch import nn
from torchmetrics import MeanAbsoluteError, PearsonCorrCoef
import unittest as ut

from graphium.trainer.predictor_summaries import SingleTaskSummary, MultiTaskSummary, STDMetric, GradientNormMetric


class test_TaskSummary(ut.TestCase):

    def test_std_metric(self):
            
        # Generate random data
        torch.random.manual_seed(42)
        rand = torch.rand(100, 1)

        # Compute expected values for STD
        expected_std = torch.std(rand, correction=0)

        # Compute std metric
        std_metric = STDMetric()
        std_metric.update(rand)
        std_metric_val = std_metric.compute()
        std_metric.reset()

        self.assertAlmostEqual(std_metric_val.item(), expected_std.item(), places=5)

        # Check multiple updates
        std_metric.update(rand[:10])
        std_metric.update(rand[10:25])
        std_metric.update(rand[25:])
        std_metric_val = std_metric.compute()
        std_metric.reset()

        self.assertAlmostEqual(std_metric_val.item(), expected_std.item(), places=5)

        # Add some correction
        expected_std = torch.std(rand, correction=1)
        std_metric = STDMetric(correction=1)
        std_metric.update(rand)
        std_metric_val = std_metric.compute()
        std_metric.reset()

        self.assertAlmostEqual(std_metric_val.item(), expected_std.item(), places=5)

        # Add some nans
        rand[[3, 5, 11, 23, 42, 56, 78, 99]] = float('nan')
        expected_std = torch.std(rand[~rand.isnan()], correction=0)

        std_metric = STDMetric(nan_strategy='ignore', correction=0)
        std_metric.update(rand)
        std_metric_val = std_metric.compute()
        std_metric.reset()

        self.assertAlmostEqual(std_metric_val.item(), expected_std.item(), places=5)
        
    def test_gradient_norm_metric(self):
        # Define a simple neural network with 2 layers
        class SimpleNN(nn.Module):
            def __init__(self):
                super(SimpleNN, self).__init__()
                torch.random.manual_seed(42)
                # Define the first layer with 10 input features and 5 output features
                self.layer1 = nn.Linear(10, 5)
                # Define the second layer with 5 input features and 1 output feature
                self.layer2 = nn.Linear(5, 1)

            def forward(self, x):
                # Pass the input through the first layer
                x = torch.relu(self.layer1(x))
                # Pass the output of the first layer through the second layer
                x = self.layer2(x)
                return x
            
        # Create an instance of the neural network, optimizer and loss function
        model = SimpleNN()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        # Generate random data
        torch.random.manual_seed(42)
        LEN = 10000
        inputs = torch.rand(LEN, 10)
        targets = torch.rand(LEN, 1)

        # Compute expected values for gradient norm
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        expected_grad_norm = torch.norm(torch.stack([torch.norm(param.grad) for param in model.parameters()]))
        optimizer.zero_grad()

        # Compute gradient norm metric
        grad_norm_metric = GradientNormMetric()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        grad_norm_metric.update(model)
        grad_norm_metric_val = grad_norm_metric.compute()
        grad_norm_metric.reset()

        self.assertAlmostEqual(grad_norm_metric_val.item(), expected_grad_norm.item(), places=5)

        # Compute gradient norm metric with many update steps
        grad_norm_metric = GradientNormMetric()
        optimizer.zero_grad()
        outputs = model(inputs[:10])
        loss = loss_fn(outputs, targets[:10])
        loss.backward()
        grad_norm_metric.update(model)
        optimizer.zero_grad()
        outputs = model(inputs[10:50])
        loss = loss_fn(outputs, targets[10:50])
        loss.backward()
        grad_norm_metric.update(model)
        optimizer.zero_grad()
        outputs = model(inputs[50:300])
        loss = loss_fn(outputs, targets[50:300])
        loss.backward()
        grad_norm_metric.update(model)
        optimizer.zero_grad()
        outputs = model(inputs[300:])
        loss = loss_fn(outputs, targets[300:])
        loss.backward()
        grad_norm_metric.update(model)

        grad_norm_metric_val = grad_norm_metric.compute()
        grad_norm_metric.reset()

        self.assertAlmostEqual(grad_norm_metric_val.item(), expected_grad_norm.item(), places=2)




if __name__ == "__main__":
    ut.main()
