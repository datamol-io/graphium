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
from copy import deepcopy
import unittest as ut

from graphium.trainer.predictor_summaries import SingleTaskSummary, MultiTaskSummary, STDMetric, GradientNormMetric

class SimpleNN(nn.Module):
# Define a simple neural network with 2 layers
    def __init__(self, in_dim=10, out_dim=1):
        super(SimpleNN, self).__init__()
        torch.random.manual_seed(42)
        # Define the first layer with 10 input features and 5 output features
        self.layer1 = nn.Linear(in_dim, 5)
        # Define the second layer with 5 input features and 1 output feature
        self.layer2 = nn.Linear(5, out_dim)

    def forward(self, x):
        # Pass the input through the first layer
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        x = torch.relu(self.layer1(x))
        # Pass the output of the first layer through the second layer
        x = self.layer2(x)
        return x
    

class SimpleDictNN(nn.Module):
    def __init__(self, task_list, in_dim=10, out_dim=1):
        super(SimpleDictNN, self).__init__()
        torch.random.manual_seed(42)
        self.dict_nn = nn.ModuleDict({task: SimpleNN(in_dim, out_dim) for task in task_list})

    def forward(self, x):
        return {task: self.dict_nn[task](x[task]) for task in self.dict_nn.keys()}


def simple_nn_grad_step(model, inputs, targets):
    # Initialize the optimizer and loss function
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a gradient step
    optimizer.zero_grad()
    outputs = model(inputs)
    if isinstance(outputs, dict):
        loss = sum([loss_fn(outputs[task], targets[task]) for task in outputs.keys()])
    else:
        loss = loss_fn(outputs, targets)
    loss.backward()
    optimizer.step()
    return model

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

        # Generate random data
        torch.random.manual_seed(42)
        LEN = 10000
        inputs = torch.rand(LEN, 10)
        targets = torch.rand(LEN, 1)

        # Compute expected values for gradient norm
        model = SimpleNN()
        model = simple_nn_grad_step(model, inputs, targets)
        expected_grad_norm = torch.norm(torch.stack([torch.norm(param.grad) for param in model.parameters()]))

        # Compute gradient norm metric
        model = SimpleNN()
        model = simple_nn_grad_step(model, inputs, targets)
        grad_norm_metric = GradientNormMetric()
        grad_norm_metric.update(model)
        grad_norm_metric_val = grad_norm_metric.compute()
        grad_norm_metric.reset()

        self.assertAlmostEqual(grad_norm_metric_val.item(), expected_grad_norm.item(), places=5)

        # Compute gradient norm metric with many update steps
        grad_norm_metric = GradientNormMetric()
        model = SimpleNN()
        model = simple_nn_grad_step(model, inputs[:50], targets[:50])
        grad_norm_metric.update(model)
        model = SimpleNN()
        model = simple_nn_grad_step(model, inputs[50:400], targets[50:400])
        grad_norm_metric.update(model)
        model = SimpleNN()
        model = simple_nn_grad_step(model, inputs[400:], targets[400:])
        grad_norm_metric.update(model)

        grad_norm_metric_val = grad_norm_metric.compute()
        grad_norm_metric.reset()

        self.assertAlmostEqual(grad_norm_metric_val.item(), expected_grad_norm.item(), places=1)

    def assertDictTensorAlmostEqual(self, dict1, dict2, places=7):
        dict1 = deepcopy(dict1)
        dict1 = {key: dict1[key] for key in sorted(dict1.keys())}
        dict2 = deepcopy(dict2)
        dict2 = {key: dict2[key] for key in sorted(dict2.keys())}
        for key in dict1.keys():
            dict1[key] = round(dict1[key].item(), places)
        for key in dict2.keys():
            dict2[key] = round(dict2[key].item(), places)
        self.assertDictEqual(dict1, dict2)


    def test_multi_task_summary(self):

        # Generate random data
        torch.random.manual_seed(42)
        targets = torch.rand(100, 3)
        preds = torch.rand(100, 3) + 0.4 * targets
        targets = {f"task{i+1}": targets[:, i] for i in range(targets.shape[1])}
        preds = {f"task{i+1}": preds[:, i] for i in range(preds.shape[1])}

        task_metrics = {
            "task1": {'mae': MeanAbsoluteError(), 'pearson': PearsonCorrCoef()},
            "task2": {'pearson': PearsonCorrCoef()},
            "task3": {'mae': MeanAbsoluteError()}
        }

        expected_dict = {}
        for task, metrics in task_metrics.items():
            for metric_name, metric in metrics.items():
                metric.update(preds[task], targets[task])
                expected_val = metric.compute()
                metric.reset()
                expected_dict[f"{task}/{metric_name}/val"] = expected_val
        

        # Test the metrics on validation step
        summary_val = MultiTaskSummary(task_metrics, step_name="val", compute_mean=False, compute_std=False)
        summary_val.update(preds, targets)
        summary_dict = summary_val.compute()
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict, places=5)

        # Test the metric reset
        summary_val.reset()
        summary_val.update(preds, targets)
        summary_dict = summary_val.compute()
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict, places=5)

        # Test multiple batches
        summary_val.reset()
        preds1 = {key: preds[key][:10] for key in preds.keys()}
        targets1 = {key: targets[key][:10] for key in targets.keys()}
        preds2 = {key: preds[key][10:25] for key in preds.keys()}
        targets2 = {key: targets[key][10:25] for key in targets.keys()}
        preds3 = {key: preds[key][25:] for key in preds.keys()}
        targets3 = {key: targets[key][25:] for key in targets.keys()}
        
        summary_val.update(preds1, targets1)
        summary_val.update(preds2, targets2)
        summary_val.update(preds3, targets3)
        summary_dict = summary_val.compute()
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict, places=5)

        # Test the mean and std computation
        summary_val = MultiTaskSummary(task_metrics, step_name="val", compute_mean=True, compute_std=True)
        summary_val.update(preds, targets)
        summary_dict = summary_val.compute()
        expected_dict_mean_std = {}
        for task in task_metrics.keys():
            expected_dict_mean_std[f"{task}/mean_preds/val"] = preds[task].mean()
            expected_dict_mean_std[f"{task}/std_preds/val"] = preds[task].std(correction=0)
            expected_dict_mean_std[f"{task}/mean_target/val"] = targets[task].mean()
            expected_dict_mean_std[f"{task}/std_target/val"] = targets[task].std(correction=0)
        expected_dict_mean_std.update(expected_dict)
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict_mean_std, places=5)

        # Test the mean and std computation with multiple batches
        summary_val.reset()
        summary_val.update(preds1, targets1)
        summary_val.update(preds2, targets2)
        summary_val.update(preds3, targets3)
        summary_dict = summary_val.compute()
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict_mean_std, places=5)

        # Test the training step doesn't return anything when no metrics on training set are selected
        summary_train = MultiTaskSummary(task_metrics, step_name="train", task_metrics_on_training_set=None, compute_mean=False, compute_std=False)
        summary_train.update(preds, targets)
        summary_train = summary_train.compute()
        self.assertDictEqual(summary_train, {})

        # Test the training step returns only the mae
        task_metrics_on_training_set = {"task1": ["mae"], "task2": None, "task3": "mae"}
        summary_train = MultiTaskSummary(task_metrics, step_name="train", task_metrics_on_training_set=task_metrics_on_training_set, compute_mean=False, compute_std=False)
        summary_train.update(preds, targets)
        summary_dict = summary_train.compute()
        expected_dict_mae = {key: value for key, value in expected_dict.items() if "mae" in key}
        expected_dict_mae = {key.replace("/val", "/train"): value for key, value in expected_dict_mae.items()}
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict_mae, places=5)

        # Test the training step returns only the mae with multiple steps
        summary_train = MultiTaskSummary(task_metrics, step_name="train", task_metrics_on_training_set=task_metrics_on_training_set, compute_mean=False, compute_std=False)
        summary_train.update(preds1, targets1)
        summary_train.update(preds2, targets2)
        summary_train.update(preds3, targets3)
        summary_dict = summary_train.compute()
        self.assertDictTensorAlmostEqual(summary_dict, expected_dict_mae, places=5)

        # Test grad_norm not available in "val" step
        summary_val = MultiTaskSummary(task_metrics, step_name="val", compute_mean=False, compute_std=False)
        summary_val.update(preds, targets)
        summary_dict = summary_val.compute()
        self.assertNotIn("grad_norm", summary_dict.keys())


if __name__ == "__main__":
    ut.main()
