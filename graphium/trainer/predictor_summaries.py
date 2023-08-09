r"""Classes to store information about resulting evaluation metrics when using a Predictor Module."""

from typing import Any, Callable, Dict, List, Optional, Union
from loguru import logger

import numpy as np
import torch
from torch import Tensor

from graphium.utils.tensor import nan_mean, nan_std, nan_median, tensor_fp16_to_fp32


class SummaryInterface(object):
    r"""
    An interface to define the functions implemented by summary classes that implement SummaryInterface.
    """

    def set_results(self, **kwargs):
        raise NotImplementedError()

    def get_dict_summary(self):
        raise NotImplementedError()

    def update_predictor_state(self, **kwargs):
        raise NotImplementedError()

    def get_metrics_logs(self, **kwargs):
        raise NotImplementedError()


class Summary(SummaryInterface):
    # TODO (Gabriela): Default argument cannot be []
    def __init__(
        self,
        loss_fun: Union[str, Callable],
        metrics: Dict[str, Callable],
        metrics_on_training_set: List[str] = [],
        metrics_on_progress_bar: List[str] = [],
        monitor: str = "loss",
        mode: str = "min",
        task_name: Optional[str] = None,
    ):
        r"""
        A container to be used by the Predictor Module that stores the results for the given metrics on the predictions and targets provided.
        Parameters:
            loss_fun:
            Loss function used during training. Acceptable strings are 'mse', 'bce', 'mae', 'cosine'.
            Otherwise, a callable object must be provided, with a method `loss_fun._get_name()`.

            metrics:
            A dictionnary of metrics to compute on the prediction, other than the loss function.
            These metrics will be logged into WandB or similar.

            metrics_on_training_set:
            The metrics names from `metrics` to be computed on the training set for each iteration.
            If `None`, all the metrics are computed. Using less metrics can significantly improve
            performance, depending on the number of readouts.

            metrics_on_progress_bar:
            The metrics names from `metrics` to display also on the progress bar of the training

            monitor:
            `str` metric to track (Default=`"loss/val"`)

            task_name:
            name of the task (Default=`None`)

        """
        self.loss_fun = loss_fun
        self.metrics = metrics
        self.metrics_on_training_set = metrics_on_training_set
        self.metrics_on_progress_bar = metrics_on_progress_bar
        self.monitor = monitor
        self.mode = mode

        self.summaries = {}
        self.best_summaries = {}

        # Current predictor state
        # self.predictor_outputs = None
        self.step_name: str = None
        self.targets: Tensor = None
        self.preds: Tensor = None
        self.loss = None  # What type?
        self.n_epochs: int = None

        self.task_name = task_name
        self.logged_metrics_exceptions = []  # Track which metric exceptions have been logged

    def update_predictor_state(
        self, step_name: str, targets: Tensor, preds: Tensor, loss: Tensor, n_epochs: int
    ):
        r"""
        update the state of the predictor
        Parameters:
            step_name: which stage you are in, e.g. "train"
            targets: the targets tensor
            predictions: the predictions tensor
            loss: the loss tensor
            n_epochs: the number of epochs
        """
        self.step_name = step_name
        self.targets = targets
        self.preds = preds
        self.loss = loss
        self.n_epochs = n_epochs

    def set_results(
        self,
        metrics: Dict[str, Tensor],
    ):
        r"""
        set the reults from the metrics
        [!] This function requires that self.update_predictor_state() be called before it.
        Parameters:
            metrics: a dictionary of metrics
        """

        # Include the task_name in the loss for logging, and similarly for other metrics
        metrics[self.metric_log_name(self.task_name, "loss", self.step_name)] = self.loss
        self.summaries[self.step_name] = Summary.Results(
            targets=self.targets,
            preds=self.preds,
            loss=self.loss,
            metrics=metrics,  # Should include task name from get_metrics_logs()
            monitored_metric=f"{self.monitor}/{self.step_name}",  # Include task name?
            n_epochs=self.n_epochs,
        )
        if self.is_best_epoch(self.step_name, self.loss, metrics):
            self.best_summaries[self.step_name] = self.summaries[self.step_name]

    def is_best_epoch(self, step_name: str, loss: Tensor, metrics: Dict[str, Tensor]) -> bool:
        r"""
        check if the current epoch is the best epoch based on self.mode criteria
        Parameters:
            step_name: which stage you are in, e.g. "train"
            loss: the loss tensor
            metrics: a dictionary of metrics
        """

        # TODO (Gabriela): Check for bugs related to monitor_name
        if not (step_name in self.best_summaries.keys()):
            return True

        # Include the task_name in the loss for logging, and similarly for other metrics
        metrics[self.metric_log_name(self.task_name, "loss", self.step_name)] = loss
        monitor_name = f"{self.monitor}/{step_name}"  # Include task_name?
        if (
            not monitor_name in self.best_summaries.keys()
        ):  # Feels like there's a bug here. What is this trying to do???
            return True

        if self.mode == "max":
            return metrics[monitor_name] > self.best_summaries[step_name].monitored
        elif self.mode == "min":
            return metrics[monitor_name] < self.best_summaries[step_name].monitored
        else:
            ValueError(f"Mode must be 'min' or 'max', provided `{self.mode}`")

    def get_results(
        self,
        step_name: str,
    ):
        r"""
        retrieve the results for a given step
        Parameters:
            step_name: which stage you are in, e.g. "train"
        Returns:
            the results for the given step
        """
        return self.summaries[step_name]

    def get_best_results(
        self,
        step_name: str,
    ):
        r"""
        retrieve the best results for a given step
        Parameters:
            step_name: which stage you are in, e.g. "train"
        Returns:
            the best results for the given step
        """
        return self.best_summaries[step_name]

    def get_results_on_progress_bar(
        self,
        step_name: str,
    ) -> Dict[str, Tensor]:
        r"""
        retrieve the results to be displayed on the progress bar for a given step
        Parameters:
            step_name: which stage you are in, e.g. "train"
        Returns:
            the results to be displayed on the progress bar for the given step
        """
        results = self.summaries[step_name]
        results_prog = {
            # f"{kk}/{step_name}": results.metrics[f"{kk}/{step_name}"] for kk in self.metrics_on_progress_bar
            self.metric_log_name(self.task_name, kk, step_name): results.metrics[
                self.metric_log_name(self.task_name, kk, step_name)
            ]
            for kk in self.metrics_on_progress_bar
        }
        return results_prog

    def get_dict_summary(self) -> Dict[str, Any]:
        r"""
        retrieve the full summary in a dictionary
        Returns:
            the full summary in a dictionary
        """
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
        Note: This function requires that self.update_predictor_state() be called before it.
        Returns:
            A dictionary of metrics to log.
        """

        targets = tensor_fp16_to_fp32(self.targets)
        preds = tensor_fp16_to_fp32(self.preds)

        targets = targets.to(dtype=preds.dtype, device=preds.device)

        # Compute the metrics always used in regression tasks
        metric_logs = {}
        metric_logs[self.metric_log_name(self.task_name, "mean_pred", self.step_name)] = nan_mean(preds)
        metric_logs[self.metric_log_name(self.task_name, "std_pred", self.step_name)] = nan_std(preds)
        metric_logs[self.metric_log_name(self.task_name, "median_pred", self.step_name)] = nan_median(preds)
        metric_logs[self.metric_log_name(self.task_name, "mean_target", self.step_name)] = nan_mean(targets)
        metric_logs[self.metric_log_name(self.task_name, "std_target", self.step_name)] = nan_std(targets)
        metric_logs[self.metric_log_name(self.task_name, "median_target", self.step_name)] = nan_median(
            targets
        )
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
            metric_name = self.metric_log_name(
                self.task_name, key, self.step_name
            )  # f"{key}/{self.step_name}"
            try:
                metric_logs[metric_name] = metric(preds, targets)
            except Exception as e:
                metric_logs[metric_name] = torch.as_tensor(float("nan"))
                # Warn only if it's the first warning for that metric
                if metric_name not in self.logged_metrics_exceptions:
                    self.logged_metrics_exceptions.append(metric_name)
                    logger.warning(f"Error for metric {metric_name}. NaN is returned. Exception: {e}")

        # Convert all metrics to CPU, except for the loss
        # metric_logs[f"{self.loss_fun._get_name()}/{self.step_name}"] = self.loss.detach().cpu()
        metric_logs[
            self.metric_log_name(self.task_name, self.loss_fun._get_name(), self.step_name)
        ] = self.loss.detach().cpu()
        # print("Metrics logs keys: ", metric_logs.keys())
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
            preds: Tensor = None,
            loss: float = None,  # Is this supposed to be a Tensor or float?
            metrics: dict = None,
            monitored_metric: str = None,
            n_epochs: int = None,
        ):
            r"""
            This inner class is used as a container for storing the results of the summary.
            Parameters:
                targets: the targets
                preds: the prediction tensor
                loss: the loss, float or tensor
                metrics: the metrics
                monitored_metric: the monitored metric
                n_epochs: the number of epochs
            """
            self.targets = targets.detach().cpu()
            self.preds = preds.detach().cpu()
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
        task_loss_fun: Callable,
        task_metrics: Dict[str, Callable],
        task_metrics_on_training_set: List[str],
        task_metrics_on_progress_bar: List[str],
        monitor: str = "loss",
        mode: str = "min",
    ):
        r"""
        class to store the summaries of the tasks
        Parameters:
            task_loss_fun: the loss function for each task
            task_metrics: the metrics for each task
            task_metrics_on_training_set: the metrics to use on the training set
            task_metrics_on_progress_bar: the metrics to use on the progress bar
            monitor: the metric to monitor
            mode: the mode of the metric to monitor
        """
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

        # Current predictor state
        self.weighted_loss = None
        self.step_name = None

    def update_predictor_state(
        self,
        step_name: str,
        targets: Dict[str, Tensor],
        preds: Dict[str, Tensor],
        loss: Tensor,
        task_losses: Dict[str, Tensor],
        n_epochs: int,
    ):
        r"""
        update the state for all predictors
        Parameters:
            step_name: the name of the step
            targets: the target tensors
            preds: the prediction tensors
            loss: the loss tensor
            task_losses: the task losses
            n_epochs: the number of epochs
        """
        self.weighted_loss = loss
        self.step_name = step_name
        for task in self.tasks:
            self.task_summaries[task].update_predictor_state(
                step_name,
                targets[task],
                preds[task].detach(),
                task_losses[task].detach(),
                n_epochs,
            )

    def set_results(self, task_metrics: Dict[str, Dict[str, Tensor]]):
        """
        set the results for all tasks
        Parameters:
            task_metrics: the metrics for each task
        """
        for task in self.tasks:
            self.task_summaries[task].set_results(task_metrics[task])
            step_name = self.task_summaries[task].step_name
            loss = self.task_summaries[task].loss
            if self.task_summaries[task].is_best_epoch(step_name, loss, task_metrics[task]):
                self.task_summaries[task].best_summaries[step_name] = self.task_summaries[task].summaries[
                    step_name
                ]

    def get_results(
        self,
        step_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        retrieve the results
        Parameters:
            step_name: the name of the step, i.e. "train"
        Returns:
            the results
        """
        results = {}
        for task in self.tasks:
            results[task] = self.task_summaries[task].get_results(step_name)
        return results

    def get_best_results(
        self,
        step_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        retrieve the best results
        Parameters:
            step_name: the name of the step, i.e. "train"
        Returns:
            the best results
        """
        results = {}
        for task in self.tasks:
            results[task] = self.task_summaries[task].get_best_results(step_name)
        return results

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
            # task_results_prog[task] = self.task_summaries[task].get_results_on_progress_bar(step_name)
            task_results_prog.update(self.task_summaries[task].get_results_on_progress_bar(step_name))
        return task_results_prog

    def get_dict_summary(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        r"""
        get task summaries in a dictionary
        Returns:
            the task summaries
        """
        task_full_dict = {}
        for task in self.tasks:
            task_full_dict[task] = self.task_summaries[task].get_dict_summary()
        return task_full_dict

    def get_metrics_logs(
        self,
    ) -> Dict[str, Dict[str, Tensor]]:
        r"""
        get the logs for the metrics
        Returns:
            the task logs for the metrics
        """
        task_metrics_logs = {}
        for task in self.tasks:
            task_metrics_logs[task] = self.task_summaries[task].get_metrics_logs()
            # average metrics
            for key in task_metrics_logs[task]:
                if isinstance(task_metrics_logs[task][key], torch.Tensor):
                    if task_metrics_logs[task][key].numel() > 1:
                        task_metrics_logs[task][key] = task_metrics_logs[task][key][
                            task_metrics_logs[task][key] != 0
                        ].mean()

        # Include global (weighted loss)
        task_metrics_logs["_global"] = {}
        task_metrics_logs["_global"][f"loss/{self.step_name}"] = self.weighted_loss.detach().cpu()
        return task_metrics_logs

    # TODO (Gabriela): This works to fix the logging on TB, but make it more efficient
    def concatenate_metrics_logs(
        self,
        metrics_logs: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Tensor]:
        r"""
        concatenate the metrics logs
        Parameters:
            metrics_logs: the metrics logs
        Returns:
            the concatenated metrics logs
        """
        concatenated_metrics_logs = {}
        for task in list(self.tasks) + ["_global"]:
            concatenated_metrics_logs.update(metrics_logs[task])
        concatenated_metrics_logs[f"loss/{self.step_name}"] = self.weighted_loss.detach().cpu()
        return concatenated_metrics_logs

    def metric_log_name(
        self,
        task_name: str,
        metric_name: str,
        step_name: str,
    ) -> str:
        r"""
        print the metric name, task name and step name
        Returns:
            the metric name, task name and step name
        """
        if task_name is None:
            return f"{metric_name}/{step_name}"
        else:
            return f"{task_name}/{metric_name}/{step_name}"
