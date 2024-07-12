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


r"""Classes to store information about resulting evaluation metrics when using a Predictor Module."""

from typing import Any, Callable, Dict, List, Optional, Union, Literal
from loguru import logger
from copy import deepcopy

import numpy as np
import torch
from torch import Tensor
from torchmetrics import MeanMetric, Metric
from torchmetrics.aggregation import BaseAggregator

from graphium.utils.tensor import nan_mean, nan_std, nan_median, tensor_fp16_to_fp32

class SummaryInterface(object):
    r"""
    An interface to define the functions implemented by summary classes that implement SummaryInterface.
    """

    def update(self, targets: Tensor, preds: Tensor) -> None:
        raise NotImplementedError()

    def compute(self, **kwargs) -> Tensor:
        raise NotImplementedError()
    
    def reset(self) -> None:
        raise NotImplementedError()


class SingleTaskSummary(SummaryInterface):
    def __init__(
        self,
        metrics: Dict[str, Union[Metric, "MetricWrapper"]],
        step_name: str,
        metrics_on_training_set: Optional[List[str]] = None,
        metrics_on_progress_bar: Optional[List[str]] = None,
        task_name: Optional[str] = None,
        compute_mean: bool = True,
        compute_std: bool = True,
    ):
        r"""
        A container to be used by the Predictor Module that stores the results for the given metrics on the predictions and targets provided.
        Parameters:

            metrics:
            A dictionnary of metrics to compute on the prediction, other than the loss function.
            These metrics will be logged into WandB or similar.

            metrics_on_training_set:
            The metrics names from `metrics` to be computed on the training set for each iteration.
            If `None`, no metrics are computed.

            metrics_on_progress_bar:
            The metrics names from `metrics` to display also on the progress bar of the training.
            If `None`, no metrics are displayed.

            task_name:
            name of the task (Default=`None`)

            compute_mean:
            whether to compute the mean of the predictions and targets

            compute_std:
            whether to compute the standard deviation of the predictions and targets

        """
        self.step_name = step_name
        if not isinstance(metrics, dict):
            raise ValueError(f"metrics must be a dictionary. Got {type(metrics)}")
        self.metrics = deepcopy(metrics)

        # Current predictor state
        # self.predictor_outputs = None
        self.task_name = task_name
        self.logged_metrics_exceptions = []  # Track which metric exceptions have been logged

        # Add default metrics
        if ("mean_pred" not in self.metrics) and compute_mean:
            self.metrics["mean_pred"] = MeanMetric(nan_strategy="ignore")
        if ("mean_target" not in self.metrics) and compute_mean:
            self.metrics["mean_target"] = MeanMetric(nan_strategy="ignore")
        if ("std_pred" not in self.metrics) and compute_std:
            self.metrics["std_pred"] = STDMetric(nan_strategy="ignore")
        if ("std_target" not in self.metrics) and compute_std:
            self.metrics["std_target"] = STDMetric(nan_strategy="ignore")

        # Parse the metrics filters
        self.metrics_on_training_set = self._parse_metrics_filter(metrics_on_training_set)
        self.metrics_on_progress_bar = self._parse_metrics_filter(metrics_on_progress_bar)

        self._cached_metrics: Dict[str, Tensor] = {}

    @property
    def get_cached_metrics(self) -> Dict[str, Tensor]:
        return deepcopy(self._cached_metrics)

    def _parse_metrics_filter(self, filter: Optional[Union[List[str], Dict[str, Any]]]) -> List[str]:
        if filter is None:
            filter = []
        elif isinstance(filter, dict):
            filter = list(filter.keys())
        elif isinstance(filter, list):
            filter = filter
        elif isinstance(filter, str):
            filter = [filter]
        else:
            raise ValueError(f"metrics_to_use must be a list or a dictionary. Got {type(filter)}")

        # Ensure that the filter is a subset of the metrics
        all_metrics = set(self.metrics.keys())
        filter = set(filter)
        if not filter.issubset(all_metrics):
            raise ValueError(f"metrics_to_use must be a subset of the metrics. Got {filter - all_metrics}, available {all_metrics}")

        return filter

    @property
    def metrics_to_use(self) -> Dict[str, Callable]:
        r"""
        return the metrics to use by filtering the metrics dictionary if it is the training step. Otherwise, return all metrics.
        """

        if self.step_name == "train":
            metrics_to_use = {
                key: metric for key, metric in self.metrics.items() if key in self.metrics_on_training_set
            }
            return metrics_to_use
        return self.metrics

    def update(self, targets: Tensor, preds: Tensor) -> None:

        r"""
        update the state of the predictor
        Parameters:
            targets: the targets tensor
            predictions: the predictions tensor
        """
        for metric_key, metric_obj in self.metrics_to_use.items():
            try:
                metric_obj.update(preds, targets)
            except:
                pass

    def _compute(self, metrics_to_use: Optional[Union[List[str], Dict[str, Any]]] = None) -> Dict[str, Tensor]:

        # Parse the metrics to use
        if metrics_to_use is None:
            metrics_to_use = list(self.metrics.keys())
        elif isinstance(metrics_to_use, dict):
            metrics_to_use = list(metrics_to_use.keys())
        else:
            raise ValueError(f"metrics_to_use must be a list or a dictionary. Got {type(metrics_to_use)}")

        # Compute the metrics
        computed_metrics = {}
        for metric_key in metrics_to_use:
            metric_name = self.metric_log_name(metric_key)
            metric_obj = self.metrics[metric_key]
            try:
                computed_metrics[f"{metric_name}"] = metric_obj.compute()
            except Exception as e:
                # If the metric computation fails, return NaN and log a warning only once
                computed_metrics[f"{metric_name}"] = torch.as_tensor(float("nan"))
                # Warn only if it's the first warning for that metric
                if metric_name not in self.logged_metrics_exceptions:
                    self.logged_metrics_exceptions.append(metric_name)
                    logger.warning(f"Error for metric {metric_name}. NaN is returned. Exception: {e}")

        return computed_metrics

    def compute(self) -> Dict[str, Tensor]:
        r"""
        compute the metrics
        Returns:
            the computed metrics
        """
        computed_metrics = self._compute(metrics_to_use=self.metrics_to_use)
        self._cached_metrics = computed_metrics

        return computed_metrics
    
    def reset(self) -> None:
        r"""
        reset the state of the metrics
        """
        for metric in self.metrics.values():
            metric.reset()

    def get_results_on_progress_bar(
        self,
    ) -> Dict[str, Tensor]:
        r"""
        retrieve the results to be displayed on the progress bar for a given step

        Returns:
            the results to be displayed on the progress bar for the given step
        """
        cached_metrics = self.get_cached_metrics
        if cached_metrics is None:
            results_prog = self._compute(metrics_to_use=self.metrics_on_progress_bar)
        else:
            results_prog = {}
            for metric_key in self.metrics_on_progress_bar:
                metric_name = self.metric_log_name(metric_key)
                results_prog[metric_name] = cached_metrics[metric_name]

        return results_prog

    def metric_log_name(self, metric_name):
        if self.task_name is None:
            return f"{metric_name}/{self.step_name}"
        else:
            return f"{self.task_name}/{metric_name}/{self.step_name}"


class MultiTaskSummary(SummaryInterface):
    def __init__(
        self,
        task_metrics: Dict[str, Dict[str, Union[Metric, "MetricWrapper"]]],
        step_name: str,
        task_metrics_on_training_set: Optional[Dict[str, List[str]]] = None,
        task_metrics_on_progress_bar: Optional[Dict[str, List[str]]] = None,
        compute_mean: bool = True,
        compute_std: bool = True,
    ):
        r"""
        class to store the summaries of the tasks
        Parameters:

        
            compute_mean:
            whether to compute the mean of the predictions and targets

            compute_std:
            whether to compute the standard deviation of the predictions and targets

        """
        self.task_metrics = task_metrics
        self.task_metrics_on_progress_bar = task_metrics_on_progress_bar if task_metrics_on_progress_bar is not None else {}
        self.task_metrics_on_training_set = task_metrics_on_training_set if task_metrics_on_training_set is not None else {}

        # Initialize all the single-task summaries
        self.tasks = list(task_metrics.keys())
        self.task_summaries: Dict[str, SingleTaskSummary] = {}
        for task in self.tasks:
            self.task_summaries[task] = SingleTaskSummary(
                metrics = self.task_metrics[task],
                step_name = step_name,
                metrics_on_training_set = self.task_metrics_on_training_set[task] if task in self.task_metrics_on_training_set else None,
                metrics_on_progress_bar = self.task_metrics_on_progress_bar[task] if task in self.task_metrics_on_progress_bar else None,
                task_name = task,
                compute_mean = compute_mean,
                compute_std = compute_std,
            )

    def update(self, targets: Dict[str, Tensor], preds: Dict[str, Tensor]) -> None:

        r"""
        update the state for all predictors
        Parameters:
            targets: the target tensors
            preds: the prediction tensors
        """
        for task in self.tasks:
            self.task_summaries[task].update(
                targets[task],
                preds[task].detach(),
            )

    def get_results_on_progress_bar(
        self,
        step_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        r"""
        return all results from all tasks for the progress bar
        Combine the dictionaries. Instead of having keys as task names, we merge all the task-specific dictionaries.
        Parameters:
            step_name: the name of the step
        Returns:
            the results for the progress bar
        """
        task_results_prog = {}
        for task in self.tasks:
            task_results_prog.update(self.task_summaries[task].get_results_on_progress_bar(step_name))
        return task_results_prog

    def compute(self) -> Dict[str, Tensor]:
        r"""
        compute the metrics for all tasks
        Returns:
            the computed metrics for all tasks
        """
        computed_metrics = {}
        for task in self.tasks:
            computed_metrics.update(self.task_summaries[task].compute())
        return computed_metrics
    
    def reset(self) -> None:
        r"""
        reset the state of the metrics
        """
        for task in self.tasks:
            self.task_summaries[task].reset()


class STDMetric(BaseAggregator):
    """
    A metric to compute the standard deviation of the predictions or targets.
    Based on `torchmetrics.Metric`, with a similar implementation to `torchmetric.MeanMetric`.

    Parameters:
        correction: 
            The correction to apply to the standard deviation. Instead of dividing by number of samples `N`,
            we divide by `N-correction`.

        nan_strategy: options:
            - ``'error'``: if any `nan` values are encountered will give a RuntimeError
            - ``'warn'``: if any `nan` values are encountered will give a warning and continue
            - ``'ignore'``: all `nan` values are silently removed
            - a float: if a float is provided will impute any `nan` values with this value

    """
    def __init__(self, nan_strategy: Union[Literal["error", "warn", "ignore"], float]="warn", correction:int=0, **kwargs):
        super().__init__(
            "sum",
            default_value=torch.tensor(0.0, dtype=torch.get_default_dtype()),
            nan_strategy=nan_strategy,
            state_name="mean_value",
            **kwargs,
        )
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_of_squares", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_weight", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.correction = correction

    def update(self, value: Union[float, Tensor], weight: Union[float, Tensor] = 1.0) -> None:
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        if not isinstance(weight, Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32)

        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        if value.numel() == 0:
            return

        self.sum += (value * weight).sum()
        self.sum_of_squares += (value * value * weight).sum()
        self.total_weight += weight.sum()

    def compute(self) -> Tensor:
        dividor = max(0, self.total_weight - self.correction)
        mean = self.sum / self.total_weight
        mean_of_squares = self.sum_of_squares / self.total_weight
        variance = mean_of_squares - mean ** 2
        variance_corr = variance * (self.total_weight / dividor)
        return torch.sqrt(variance_corr)

class GradientNormMetric(Metric):
    """
    A metric to compute the norm of the gradient.
    Based on `torchmetrics.Metric`.

    Warning:
        This metric is not compatible with other metrics since it doesn't take
        the predictions and targets as input. It takes the model as input.
    """
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("gradient_norm_sq", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_steps", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, model: torch.nn.Module) -> None:
        total_norm = torch.tensor(0.0)
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.detach().cpu() ** 2
        self.gradient_norm_sq += total_norm
        self.total_steps += 1

    def compute(self) -> Tensor:
        return (self.gradient_norm_sq / self.total_steps).sqrt()
