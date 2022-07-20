r"""Classes to store information about resulting evaluation metrics when using a Predictor Module."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
from mordred import Result

import numpy as np
import torch
from torch import Tensor

from goli.utils.tensor import nan_mean, nan_std, nan_median

class SummaryInterface(object):
    r"""An interface to define the functions implemented by summary classes that implement SummaryInterface."""
    def set_results(self, **kwargs):
        raise NotImplementedError()

    def get_dict_summary(self):
        raise NotImplementedError()

    def update_predictor_state(self, **kwargs):
        raise NotImplementedError()

    def get_metrics_logs(self, **kwargs):
        raise NotImplementedError()

class Summary(SummaryInterface):
    r"""A container to be used by the Predictor Module that stores the results for the given metrics on the predictions and targets provided."""
    #TODO (Gabriela): Default argument cannot be []
    def __init__(
        self, 
        loss_fun, 
        metrics, 
        metrics_on_training_set, 
        metrics_on_progress_bar=[], 
        monitor="loss", 
        mode: str = "min",
        task_name: Optional[str] = None,
    ):
        self.loss_fun = loss_fun
        self.metrics = metrics
        self.metrics_on_training_set = metrics_on_training_set
        self.metrics_on_progress_bar = metrics_on_progress_bar
        self.monitor = monitor
        self.mode = mode

        self.summaries = {}
        self.best_summaries = {}

        # Current predictor state
        #self.predictor_outputs = None
        self.step_name: str = None
        self.targets: Tensor = None
        self.predictions: Tensor = None
        self.loss = None                            # What type?
        self.n_epochs: int = None

        self.task_name = task_name

    def update_predictor_state(self, step_name, targets, predictions, loss, n_epochs):
        self.step_name = step_name
        self.targets = targets
        self.predictions = predictions
        self.loss = loss
        self.n_epochs = n_epochs

    def set_results(self, metrics) -> float:
        r"""This function requires that self.update_predictor_state() be called before it."""
        
        # Include the task_name in the loss for tensorboard, and similarly for other metrics
        metrics[self.metric_log_name(self.task_name, "loss", self.step_name)] = self.loss
        self.summaries[self.step_name] = Summary.Results(
            targets=self.targets,
            predictions=self.predictions,
            loss=self.loss,
            metrics=metrics,                                                    # Should include task name from get_metrics_logs()
            monitored_metric=f"{self.monitor}/{self.step_name}",                # Include task name?
            n_epochs=self.n_epochs
        )
        if self.is_best_epoch(self.step_name, self.loss, metrics):
            self.best_summaries[self.step_name] = self.summaries[self.step_name]

    def is_best_epoch(self, step_name, loss, metrics):
        """TODO (Gabriela): Check for bugs related to monitor_name"""
        if not (step_name in self.best_summaries.keys()):
            return True

        # Include the task_name in the loss for tensorboard, and similarly for other metrics
        metrics[self.metric_log_name(self.task_name, "loss", self.step_name)] = loss
        monitor_name = f"{self.monitor}/{step_name}"            # Include task_name?
        if not monitor_name in self.best_summaries.keys():      # Feels like there's a bug here. What is this trying to do???
            return True

        if self.mode == "max":
            return metrics[monitor_name] > self.best_summaries[step_name].monitored
        elif self.mode == "min":
            return metrics[monitor_name] < self.best_summaries[step_name].monitored
        else:
            ValueError(f"Mode must be 'min' or 'max', provided `{self.mode}`")

    def get_results(self, step_name):
        return self.summaries[step_name]

    def get_best_results(self, step_name):
        return self.best_summaries[step_name]

    def get_results_on_progress_bar(self, step_name):
        results = self.summaries[step_name]
        results_prog = {
            #f"{kk}/{step_name}": results.metrics[f"{kk}/{step_name}"] for kk in self.metrics_on_progress_bar
            self.metric_log_name(self.task_name, kk, step_name): results.metrics[self.metric_log_name(self.task_name, kk, step_name)] for kk in self.metrics_on_progress_bar
        }
        return results_prog

    def get_dict_summary(self):
        full_dict = {}
        # Get metric summaries
        full_dict["metric_summaries"] = {}
        for key, val in self.summaries.items():
            full_dict["metric_summaries"][key] = {k: v for k, v in val.metrics.items()}
            full_dict["metric_summaries"][key]["n_epochs"] = val.n_epochs

        # Get metric summaries at best epoch
        full_dict["best_epoch_metric_summaries"] = {}
        for key, val in self.best_summaries.items():
            full_dict["best_epoch_metric_summaries"][key] = val.metrics
            full_dict["best_epoch_metric_summaries"][key]["n_epochs"] = val.n_epochs

        return full_dict

    def get_metrics_logs(self) -> Dict[str, Any]:
        r"""
        Get the data about metrics to log.
        
        Note: This function requires that self.update_predictor_state() be called before it."""
        targets = self.targets.to(dtype=self.predictions.dtype, device=self.predictions.device)

        # Compute the metrics always used in regression tasks
        metric_logs = {}
        metric_logs[self.metric_log_name(self.task_name, "mean_pred", self.step_name)] = nan_mean(self.predictions)
        metric_logs[self.metric_log_name(self.task_name, "std_pred", self.step_name)] = nan_std(self.predictions)
        metric_logs[self.metric_log_name(self.task_name, "median_pred", self.step_name)] = nan_median(self.predictions)
        metric_logs[self.metric_log_name(self.task_name, "mean_target", self.step_name)] = nan_mean(targets)
        metric_logs[self.metric_log_name(self.task_name, "std_target", self.step_name)] = nan_std(targets)
        metric_logs[self.metric_log_name(self.task_name, "median_target", self.step_name)] = nan_median(targets)
        if torch.cuda.is_available():
            metric_logs[f"gpu_allocated_GB"] = torch.tensor(torch.cuda.memory_allocated() / (2**30))

        # Specify which metrics to use
        metrics_to_use = self.metrics
        if self.step_name == "train":
            metrics_to_use = {
                key: metric for key, metric in metrics_to_use.items() if key in self.metrics_on_training_set
            }

        # Compute the additional metrics
        for key, metric in metrics_to_use.items():
            metric_name = self.metric_log_name(self.task_name, key, self.step_name)   #f"{key}/{self.step_name}"
            try:
                metric_logs[metric_name] = metric(self.predictions, targets)
            except Exception as e:
                metric_logs[metric_name] = torch.as_tensor(float("nan"))

        # Convert all metrics to CPU, except for the loss
        #metric_logs[f"{self.loss_fun._get_name()}/{self.step_name}"] = self.loss.detach().cpu()
        metric_logs[self.metric_log_name(self.task_name, self.loss_fun._get_name(), self.step_name)] = self.loss.detach().cpu()
        #print("Metrics logs keys: ", metric_logs.keys())
        metric_logs = {key: metric.detach().cpu() for key, metric in metric_logs.items()}

        return metric_logs

    def metric_log_name(self, task_name, metric_name, step_name):
        if task_name is None:
            return f"{metric_name}/{step_name}"
        else:
            return f"{task_name}/{metric_name}/{step_name}"

    class Results:
        def __init__(
            self,
            targets: Tensor = None,
            predictions: Tensor = None,
            loss: float = None,                 # Is this supposed to be a Tensor or float?
            metrics: dict = None,
            monitored_metric: str = None,
            n_epochs: int = None,
        ):
            r"""This inner class is used as a container for storing the results of the summary."""
            self.targets = targets.detach().cpu()
            self.predictions = predictions.detach().cpu()
            self.loss = loss.item() if isinstance(loss, Tensor) else loss
            self.monitored_metric = monitored_metric
            if monitored_metric in metrics.keys():
                self.monitored = metrics[monitored_metric].detach().cpu()
            self.metrics = {
                key: value.tolist() if isinstance(value, (Tensor, np.ndarray)) else value
                for key, value in metrics.items()
            }
            self.n_epochs = n_epochs


class TaskSummaries(SummaryInterface):
    def __init__(
        self, 
        task_loss_fun, 
        task_metrics, 
        task_metrics_on_training_set, 
        task_metrics_on_progress_bar, 
        monitor="loss", 
        mode: str = "min"
    ):
        self.task_loss_fun = task_loss_fun
        self.task_metrics = task_metrics
        self.task_metrics_on_progress_bar = task_metrics_on_progress_bar
        self.task_metrics_on_training_set = task_metrics_on_training_set
        self.monitor = monitor
        self.mode = mode

        self.task_summaries: Dict[str, Summary] = {}
        self.task_best_summaries: Dict[str, Summary] = {}
        self.tasks = list(task_loss_fun.keys())

        for task in self.tasks:
            self.task_summaries[task] = Summary(
                self.task_loss_fun[task],
                self.task_metrics[task],
                self.task_metrics_on_training_set[task],
                self.task_metrics_on_progress_bar[task],
                self.monitor,
                self.mode,
                task_name=task,
            )

    def update_predictor_state(self, step_name, targets, predictions, loss, n_epochs):
        for task in self.tasks:
            self.task_summaries[task].update_predictor_state(
                step_name,
                targets[task],
                predictions[task],
                loss,
                n_epochs,
            )

    def set_results(self, task_metrics):
        for task in self.tasks:
            self.task_summaries[task].set_results(task_metrics[task])
            step_name = self.task_summaries[task].step_name
            loss = self.task_summaries[task].loss
            if self.task_summaries[task].is_best_epoch(step_name, loss, task_metrics[task]):
                self.task_summaries[task].best_summaries[step_name] = self.task_summaries[task].summaries[step_name]

    def get_results(self, step_name):
        results = {}
        for task in self.tasks:
            results[task] = self.task_summaries[task].get_results(step_name)
        return results
    
    def get_best_results(self, step_name):
        results = {}
        for task in self.tasks:
            results[task] = self.task_summaries[task].get_best_results(step_name)
        return results

    def get_results_on_progress_bar(self, step_name):
        task_results_prog = {}
        for task in self.tasks:
            task_results_prog[task] = self.task_summaries[task].get_results_on_progress_bar(step_name)
        return task_results_prog

    def get_dict_summary(self):
        task_full_dict = {}
        for task in self.tasks:
            task_full_dict[task] = self.task_summaries[task].get_dict_summary()
        return task_full_dict

    def get_metrics_logs(self):
        task_metrics_logs = {}
        for task in self.tasks:
            task_metrics_logs[task] = self.task_summaries[task].get_metrics_logs()
        return task_metrics_logs

    def metric_log_name(self, task_name, metric_name, step_name):
        if task_name is None:
            return f"{metric_name}/{step_name}"
        else:
            return f"{task_name}/{metric_name}/{step_name}"