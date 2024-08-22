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


import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal, Mapping

import lightning
import numpy as np
import torch
from loguru import logger
from mup.optim import MuAdam
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torchmetrics import Metric

from graphium.config.config_convert import recursive_config_reformating
from graphium.data.datamodule import BaseDataModule
from graphium.trainer.metrics import MetricWrapper, LossWrapper
from graphium.trainer.predictor_options import (
    EvalOptions,
    FlagOptions,
    ModelOptions,
    OptimOptions,
)
from graphium.trainer.predictor_summaries import MultiTaskSummary, GradientNormMetric
from graphium.utils import fs
from graphium.utils.moving_average_tracker import MovingAverageTracker
from graphium.utils.tensor import dict_tensor_fp16_to_fp32


class PredictorModule(lightning.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        loss_fun: Dict[str, Union[str, Callable]],
        task_levels: Dict[str, str],
        random_seed: int = 42,
        featurization: Dict[str, str] = None,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        torch_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Optional[Union[str, int]] = None,
        multitask_handling: Optional[str] = None,
        metrics: Dict[str, Dict[str, Union[Metric, "MetricWrapper"]]] = None,
        metrics_on_progress_bar: Dict[str, List[str]] = [],
        metrics_on_training_set: Optional[Dict[str, List[str]]] = None,
        flag_kwargs: Dict[str, Any] = None,
        task_norms: Optional[Dict[Callable, Any]] = None,
        metrics_every_n_train_steps: Optional[int] = None,
        replicas: int = 1,
        gradient_acc: int = 1,
        global_bs: Optional[int] = 1,
    ):
        """
        The Lightning module responsible for handling the predictions, losses, metrics, optimization, etc.
        It works in a multi-task setting, with different losses and metrics per class

        Parameters:
            model_class: The torch Module containing the main forward function
            model_kwargs: The arguments to initialize the model from `model_class`
            loss_fun: A `dict[str, fun]`, where `str` is the task name and `fun` the loss function
            random_seed: The seed for random initialization
            optim_kwargs: The optimization arguments. See class `OptimOptions`
            torch_scheduler_kwargs: The torch scheduler arguments. See class `OptimOptions`
            scheduler_kwargs: The lightning scheduler arguments. See class `OptimOptions`
            target_nan_mask: How to handle the NaNs. See `MetricsWrapper` for options
            metrics: A `dict[str, fun]`, where `str` is the task name and `fun` the metric function
            metrics_on_progress_bar: A `dict[str, list[str2]`, where `str` is the task name and `str2` the metrics to include on the progress bar
            metrics_on_training_set: A `dict[str, list[str2]`, where `str` is the task name and `str2` the metrics to include on the training set
            flag_kwargs: Arguments related to using the FLAG adversarial augmentation
            task_norms: the normalization for each task
            metrics_every_n_train_steps: Compute and log metrics every n training steps.
                Set to `None` to never log the training metrics and statistics (the default). Set to `1` to log at every step.
        """
        self.save_hyperparameters()

        self.random_seed = random_seed
        torch.random.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.target_nan_mask = target_nan_mask
        self.multitask_handling = multitask_handling
        self.task_levels = task_levels
        self.featurization = featurization
        self.task_norms = task_norms

        super().__init__()

        # Setting the model options
        self.model_kwargs = model_kwargs
        self._model_options = ModelOptions(model_class=model_class, model_kwargs=model_kwargs)
        # Setting the optimizer options
        self.optim_options = OptimOptions(
            optim_kwargs=optim_kwargs,
            torch_scheduler_kwargs=torch_scheduler_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        # Setting the evaluation options
        eval_options = {}
        for task in loss_fun:
            eval_options[task] = EvalOptions(
                loss_fun=loss_fun[task],
                metrics=metrics[task],
                metrics_on_progress_bar=metrics_on_progress_bar[task],
                metrics_on_training_set=metrics_on_training_set[task]
                if metrics_on_training_set is not None
                else None,
            )
            eval_options[task].check_metrics_validity()

        self._eval_options_dict: Dict[str, EvalOptions] = eval_options
        self._eval_options_dict = {
            self._get_task_key(task_level=task_levels[key], task=key): value
            for key, value in self._eval_options_dict.items()
        }
        # Setting the flag options
        self._flag_options = FlagOptions(flag_kwargs=flag_kwargs)

        self.model = self._model_options.model_class(**self._model_options.model_kwargs)

        loss_fun = {
            self._get_task_key(task_level=task_levels[key], task=key): value
            for key, value in loss_fun.items()
        }
        self.tasks = list(loss_fun.keys())

        # Task-specific evalutation attributes
        self.loss_fun = {}
        loss_names = {}
        self.metrics = {}
        self.metrics_on_progress_bar = {}
        self.metrics_on_training_set = {}
        for task in self.tasks:
            loss_names[task], self.loss_fun[task] = EvalOptions.parse_loss_fun(loss_fun[task])
            self.metrics[task] = (
                self._eval_options_dict[task].metrics
                if self._eval_options_dict[task].metrics is not None
                else {}
            )
            self.metrics_on_progress_bar[task] = self._eval_options_dict[task].metrics_on_progress_bar
            self.metrics_on_training_set[task] = (
                list(self.metrics[task].keys())
                if self._eval_options_dict[task].metrics_on_training_set is None
                else self._eval_options_dict[task].metrics_on_training_set
            )
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Set the parameters and default values for the FLAG adversarial augmentation, and check values
        self._flag_options.set_kwargs()
        self.flag_kwargs = self._flag_options.flag_kwargs

        # Set the parameters for optimizer options
        self.optim_options.set_kwargs()

        # Add the loss to the metrics
        metrics_with_loss = deepcopy(self.metrics)
        for task in self.tasks:
            metrics_with_loss[task][f"loss_{loss_names[task]}"] = MetricWrapper(
                metric=LossWrapper(self.loss_fun[task]),
                target_nan_mask=self.target_nan_mask,
                multitask_handling=self.multitask_handling,
            )
        
        # Initialize the epoch summary
        self.task_epoch_summary = {
            "train": MultiTaskSummary(
                task_metrics=metrics_with_loss, 
                step_name="train", 
                task_metrics_on_progress_bar=None,
                task_metrics_on_training_set=self.metrics_on_training_set,
                ),
            "val": MultiTaskSummary(
                task_metrics=metrics_with_loss, 
                step_name="val", 
                task_metrics_on_progress_bar=self.metrics_on_progress_bar,
                task_metrics_on_training_set=None,
                ),
            "test": MultiTaskSummary(
                task_metrics=metrics_with_loss,
                step_name="test",
                task_metrics_on_progress_bar=None,
                task_metrics_on_training_set=None,
            ),
        }

        # This helps avoid a bug when saving hparams to yaml with different dict or str formats
        self._set_hparams(recursive_config_reformating(self.hparams))

        # throughput estimation
        self.mean_time_tracker = MovingAverageTracker()
        self.mean_tput_tracker = MovingAverageTracker()
        self.epoch_start_time = None

        # Decide whether to log every step or once at the end
        # of the epoch.
        self.metrics_every_n_train_steps = metrics_every_n_train_steps
        # Wether save preds and targets for each training step.

        self.samples_seen = 0
        self.global_bs = global_bs
        self.model_grad = GradientNormMetric()

    def forward(
        self, inputs: Dict
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor], Dict[str, Dict[str, Tensor]]]]:
        r"""
        Returns the result of `self.model.forward(*inputs)` on the inputs.
        If the output of `out = self.model.forward` is a dictionary with a `"preds"` key,
        it is returned directly. Otherwise, a new dictionary is created and
        returns `{"preds": out}`.

        Returns:
            A dict with a key `"preds"` representing the prediction of the network.

        """
        # Convert to the right dtype and run the model
        feats = self._convert_features_dtype(inputs["features"])
        # *check for nan in model output
        out = self.model.forward(feats)
        if isinstance(out, dict) and ("preds" in out.keys()):
            out_dict = out
        else:
            out_dict = {"preds": out}

        return out_dict

    def _convert_features_dtype(self, feats):
        # Convert features to dtype
        if isinstance(feats, torch.Tensor):
            feats = feats.to(self.dtype)
        elif isinstance(feats, (Data, Batch, dict)):
            for key, val in feats.items():
                if isinstance(val, torch.Tensor) and (val.is_floating_point()):
                    feats[key] = val.to(dtype=self.dtype)
        return feats

    def _get_task_key(self, task_level: str, task: str):
        task_prefix = f"{task_level}_"
        if not task.startswith(task_prefix):
            task = task_prefix + task
        return task
    
    def _get_average_loss_from_outputs(self, outputs: Dict[Literal["loss", "task_losses"], Tensor], step_name: Literal["train", "val", "test"]) -> Dict[str, Tensor]:
        r"""
        Averages the loss over the different tasks
        """
        global_loss = torch.as_tensor(outputs["loss"]).detach()
        if global_loss.numel() > 1:
            global_loss = global_loss[global_loss != 0].mean()
        average_losses = {f"_global/loss/{step_name}": global_loss}
        for task in self.tasks:
            this_losses = torch.as_tensor(outputs["task_losses"][task]).detach()
            if this_losses.numel() > 1:
                this_losses = this_losses[this_losses != 0].mean()
            average_losses[f"{task}/loss/{step_name}"] = this_losses
        return average_losses


    def configure_optimizers(self, impl=None):
        if impl is None:
            impl = torch.optim.Adam

        # Define the optimizer and schedulers
        optimiser = MuAdam(self.parameters(), **self.optim_options.optim_kwargs, impl=impl)
        self.optim_options.torch_scheduler_kwargs.pop("module_type")
        torch_scheduler = self.optim_options.scheduler_class(
            optimizer=optimiser, **self.optim_options.torch_scheduler_kwargs
        )
        scheduler = {
            "scheduler": torch_scheduler,
            **self.optim_options.scheduler_kwargs,
        }
        return [optimiser], [scheduler]

    @staticmethod
    def compute_loss(
        preds: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        weights: Optional[Tensor],
        loss_fun: Dict[str, Callable],
        target_nan_mask: Optional[Union[str, int]] = None,
        multitask_handling: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        r"""
        Compute the loss using the specified loss function, and dealing with
        the nans in the `targets`.

        Parameters:
            preds:
                Predicted values

            targets:
                Target values

            weights:
                No longer supported, will raise an error.

            target_nan_mask:

                - None: Do not change behaviour if there are NaNs

                - int, float: Value used to replace NaNs. For example, if `target_nan_mask==0`, then
                  all NaNs will be replaced by zeros

                - 'ignore': The NaN values will be removed from the tensor before computing the metrics.
                  Must be coupled with the `multitask_handling='flatten'` or `multitask_handling='mean-per-label'`.

            multitask_handling:
                - None: Do not process the tensor before passing it to the metric.
                  Cannot use the option `multitask_handling=None` when `target_nan_mask=ignore`.
                  Use either 'flatten' or 'mean-per-label'.

                - 'flatten': Flatten the tensor to produce the equivalent of a single task

                - 'mean-per-label': Loop all the labels columns, process them as a single task,
                    and average the results over each task
                  *This option might slowdown the computation if there are too many labels*

            loss_fun:
                Loss function to use

        Returns:
            Tensor:
                weighted_loss: Resulting weighted loss
                all_task_losses: Loss per task
        """

        wrapped_loss_fun_dict = {
            task: MetricWrapper(
                metric=LossWrapper(loss),
                threshold_kwargs=None,
                target_nan_mask=target_nan_mask,
                multitask_handling=multitask_handling,
            )
            for task, loss in loss_fun.items()
        }

        if weights is not None:
            raise NotImplementedError("Weights are no longer supported in the loss")

        all_task_losses = {
            task: wrapped.update_compute(preds=preds[task], target=targets[task])
            for task, wrapped in wrapped_loss_fun_dict.items()
        }

        total_loss = torch.sum(torch.stack(list(all_task_losses.values())), dim=0)
        num_tasks = len(all_task_losses.keys())
        weighted_loss = total_loss / num_tasks
        return weighted_loss, all_task_losses

    def _general_step(self, batch: Dict[str, Tensor], step_name: Literal["train", "val", "test"]) -> Dict[str, Any]:
        r"""Common code for training_step, validation_step and testing_step"""
        preds = self.forward(batch)  # The dictionary of predictions

        # * check for nan in model output
        targets_dict = batch.get("labels")

        # Different type of preds can be return by the forward
        if isinstance(preds, dict) and ("preds" in preds.keys()):
            preds = preds["preds"]
        elif isinstance(preds, Tensor):
            preds = {k: preds[ii] for ii, k in enumerate(targets_dict.keys())}

        preds = {
            self._get_task_key(task_level=self.task_levels[key], task=key): value
            for key, value in preds.items()
        }
        # preds = {k: preds[ii] for ii, k in enumerate(targets_dict.keys())}
        for task, pred in preds.items():
            task_specific_norm = self.task_norms[task] if self.task_norms is not None else None
            if hasattr(task_specific_norm, "normalize_val_test"):
                normalize_val_test = task_specific_norm.normalize_val_test
            else:
                normalize_val_test = False
            if step_name != "train" and not normalize_val_test:
                # apply denormalization for val and test predictions for correct loss and metrics evaluation
                # if normalize_val_test is not true, only train loss will stay as the normalized version
                # if normalize_val_test is true, no denormalization is applied, all losses and metrics are normalized version
                preds[task] = task_specific_norm.denormalize(pred)
            targets_dict[task] = targets_dict[task].to(dtype=pred.dtype)
        weights = batch.get("weights", None)

        loss, task_losses = self.compute_loss(
            preds=preds,
            targets=targets_dict,
            weights=weights,
            loss_fun=self.loss_fun,
            target_nan_mask=self.target_nan_mask,
            multitask_handling=self.multitask_handling,
        )

        for task in preds:
            task_specific_norm = self.task_norms[task] if self.task_norms is not None else None
            if hasattr(task_specific_norm, "normalize_val_test"):
                normalize_val_test = task_specific_norm.normalize_val_test
            else:
                normalize_val_test = False
            if step_name == "train" and not normalize_val_test:
                # apply denormalization for targets and predictions for the evaluation of training metrics (excluding loss)
                # if normalize_val_test is not true, train loss will stay as the normalized version
                # if normalize_val_test is true, no denormalization is applied, all losses and metrics are normalized version
                preds[task] = task_specific_norm.denormalize(preds[task])
                targets_dict[task] = task_specific_norm.denormalize(targets_dict[task])
            preds[task] = preds[task].detach()
            targets_dict[task] = targets_dict[task].detach()

        self.task_epoch_summary[step_name].update(preds, targets_dict)

        step_dict = {}
        step_dict["loss"] = loss
        step_dict["task_losses"] = task_losses
        return step_dict


    def flag_step(self, batch: Dict[str, Tensor], step_name: Literal["train", "val", "test"]) -> Dict[str, Any]:
        r"""
        Perform adversarial data agumentation during one training step using FLAG.
        Paper: https://arxiv.org/abs/2010.09891
        Github: https://github.com/devnkong/FLAG
        """

        alpha, n_steps = self.flag_kwargs["alpha"], self.flag_kwargs["n_steps"]

        X = self._convert_features_dtype(batch["features"])
        X_shape = X["feat"].shape

        pert = torch.FloatTensor(X_shape).uniform_(-alpha, alpha).to(device=X["feat"].device)
        pert.requires_grad = True

        # Perturb the features
        pert_batch = deepcopy(batch)
        features = pert_batch["features"]
        features["feat"] = features["feat"] + pert

        preds = self.forward(pert_batch)["preds"]
        targets = batch.pop("labels")
        for key in targets.keys():
            targets[key] = targets[key].to(dtype=preds[key].dtype)
        weights = batch.pop("weights", None)
        loss, task_losses = self.compute_loss(
            preds=preds,
            targets=targets,
            weights=weights,
            target_nan_mask=self.target_nan_mask,
            multitask_handling=self.multitask_handling,
            loss_fun=self.loss_fun,
        )

        loss = loss / n_steps

        # Iteratively augment data by applying perturbations
        # Accumulate the gradients to be applied to the weights of the network later on
        for _ in range(n_steps - 1):
            loss.backward()
            pert_data = pert.detach() + alpha * torch.sign(pert.grad.detach())
            pert.data = pert_data.data
            pert.grad[:] = 0
            features["feat"] = features["feat"] + pert
            pert_batch["features"] = features
            preds = self.forward(pert_batch)["preds"]
            loss, _ = self.compute_loss(
                preds=preds,
                targets=targets,
                weights=weights,
                target_nan_mask=self.target_nan_mask,
                multitask_handling=self.multitask_handling,
                loss_fun=self.loss_fun,
            )
            loss = loss / n_steps

        for key in preds.keys():
            preds[key] = preds[key].detach()
            targets[key] = targets[key].detach()
        if weights is not None:
            weights = weights.detach()

        step_dict = {}
        step_dict[f"loss/{step_name}"] = loss.detach().cpu()
        step_dict["loss"] = loss
        step_dict["task_losses"] = task_losses
        self.task_epoch_summary[step_name].update(preds, targets)
        return step_dict

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:

        self.model_grad.reset()
        self.task_epoch_summary["train"].reset()
        self.batch_start_time = time.time()
        self.skip_log_train_metrics = (self.metrics_every_n_train_steps is None) or (
            (batch_idx % self.metrics_every_n_train_steps) != 0
        )
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        train_batch_time = time.time() - self.batch_start_time  # To be used for throughput calculation

        # Get the metrics that are logged at every step (loss, grad_norm, batch_time, batch_tput)
        metrics_logs = {}
        # Incriment by the batch size
        self.samples_seen += self.global_bs
        metrics_logs["samples_seen"] = self.samples_seen

        # report the training loss for each individual tasks
        # get the mean loss value for individual tasks as they are a tensor of size --> gradient accumulation * replication * device_iter
        # filter zeros out for the individual losses
        losses = self._get_average_loss_from_outputs(outputs, step_name="train")

        metrics_logs.update(losses)
        metrics_logs.update(self.task_epoch_summary["train"].compute())

        # If logging is skipped for this step, then log the important metrics anyway and return
        if self.skip_log_train_metrics:
            self.log_dict(
                dictionary=metrics_logs,
                logger=True,
                on_step=True,
                prog_bar=True,
            )
            return

        ### The code below is not executed if the logging is skipped for this step ###

        # Get the throughput of the batch
        num_graphs = self.get_num_graphs(batch["features"])
        tput = num_graphs / train_batch_time
        metrics_logs["_global/train/batch_time"] = train_batch_time
        metrics_logs["_global/train/batch_tput"] = tput
        self.mean_time_tracker.update(train_batch_time)
        self.mean_tput_tracker.update(tput)

        metrics_computed = self.task_epoch_summary["train"].compute()
        self.task_epoch_summary["train"].reset()
        metrics_logs.update(metrics_computed)
        metrics_logs["_global/train/grad_norm"] = self.model_grad.compute()

        # Log the metrics
        self.log_dict(
            dictionary=metrics_logs,
            logger=True,
            on_step=True,
            prog_bar=True,
        )

    def training_step(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        step_dict = None

        # Train using FLAG
        if self.flag_kwargs["n_steps"] > 0:
            step_dict = self.flag_step(batch=batch, step_name="train")
        # Train normally, without using FLAG
        elif self.flag_kwargs["n_steps"] == 0:
            # step_dict = self._general_step(batch=batch, step_name="train")
            step_dict = self._general_step(batch=batch, step_name="train")

        # Update the gradients
        self.model_grad.update(self.model)

        return step_dict  # Returning the metrics_logs with the loss


    def validation_step(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        return self._general_step(batch=batch, step_name="val")

    def test_step(self, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        return self._general_step(batch=batch, step_name="test")
    
    def _general_epoch_start(self, step_name: Literal["train", "val", "test"]) -> None:
        self.task_epoch_summary[step_name].reset()
        self.epoch_start_time = time.time()
        self.mean_time_tracker.reset()
        self.mean_tput_tracker.reset()


    def _general_epoch_end(self, step_name: Literal["train", "val", "test"]) -> Dict[str, Tensor]:
        r"""Common code for training_epoch_end, validation_epoch_end and testing_epoch_end"""
        # Transform the list of dict of dict, into a dict of list of dict
        
        metric_logs = self.task_epoch_summary[step_name].compute()
        self.task_epoch_summary[step_name].reset()
        metric_logs_cpu = {k: v for k, v in metric_logs.items() if v.device == torch.device("cpu")}
        if len(metric_logs_cpu) > 0:
            self.log_dict(metric_logs_cpu, logger=True, prog_bar=True, sync_dist=False, on_epoch=True)
        
        metric_logs_accelerator = {k: v for k, v in metric_logs.items() if v.device != torch.device("cpu")}
        if len(metric_logs_accelerator) > 0:
            self.log_dict(metric_logs_accelerator, logger=True, prog_bar=True, sync_dist=True, on_epoch=True)

        # Time metrics are tracked always on CPU, without progress bar, so we log them separatly
        time_metrics = {}
        time_metrics[f"_global/{step_name}/mean_batch_time"] = torch.tensor(self.mean_time_tracker.mean_value)
        time_metrics[f"_global/{step_name}/mean_tput"] = self.mean_tput_tracker.mean_value
        time_metrics[f"_global/{step_name}/epoch_time"] = torch.tensor(time.time() - self.epoch_start_time)

        self.log_dict(time_metrics, logger=True, prog_bar=False, sync_dist=False, on_epoch=True)

        return metric_logs

    def on_train_epoch_start(self) -> None:
        self._general_epoch_start(step_name="train")

    def on_train_epoch_end(self) -> None:
        self._general_epoch_end(step_name="train")

    def on_validation_epoch_start(self) -> None:
        self._general_epoch_start(step_name="val")
        return super().on_validation_epoch_start()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.batch_start_time = time.time()
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        val_batch_time = time.time() - self.batch_start_time
        self.mean_time_tracker.update(val_batch_time)
        num_graphs = self.get_num_graphs(batch["features"])
        self.mean_tput_tracker.update(num_graphs / val_batch_time)
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        self._general_epoch_end(step_name="val")
        return super().on_validation_epoch_end()


    def on_test_epoch_start(self) -> None:
        self._general_epoch_start(step_name="test")
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        self._general_epoch_end(step_name="test")
        return super().on_test_epoch_end()
    
    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        self.batch_start_time = time.time()
        return super().on_test_batch_start(batch, batch_idx, dataloader_idx)
    
    def on_test_batch_end(self, outputs: Tensor | Mapping[str, Any] | None, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        test_batch_time = time.time() - self.batch_start_time
        self.mean_time_tracker.update(test_batch_time)
        num_graphs = self.get_num_graphs(batch["features"])
        self.mean_tput_tracker.update(num_graphs / test_batch_time)
        return super().on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_train_start(self):
        hparams_log = deepcopy(self.hparams)
        hparams_log["n_params"] = self.n_params
        if self.logger is not None:
            self.logger.log_hyperparams(hparams_log)

    @property
    def get_metrics_on_progress_bar(self) -> List[str]:
        prog_list = ["_global/loss/train"]
        for task_name in self.tasks:
            for metric in self.metrics_on_progress_bar[task_name]:
                this_summary = self.task_epoch_summary["val"][task_name]
                prog_list.append(this_summary.metric_log_name(metric))

        return prog_list

    def __repr__(self) -> str:
        r"""
        Controls how the class is printed
        """
        model_str = self.model.__repr__()
        summary_str = self.summarize().__repr__()

        return model_str + "\n\n" + summary_str

    @staticmethod
    def list_pretrained_models():
        """List available pretrained models."""
        from graphium.utils.spaces import GRAPHIUM_PRETRAINED_MODELS_DICT

        return GRAPHIUM_PRETRAINED_MODELS_DICT # Avoiding circular imports with `space.py`

    @staticmethod
    def load_pretrained_model(name_or_path: str, device: str = None, strict: bool = True, **kwargs):
        """Load a pretrained model from its name.

        Args:
            name: Name of the model to load or a valid checkpoint path. List available
                from `graphium.trainer.PredictorModule.list_pretrained_models()`.
        """

        from graphium.utils.spaces import GRAPHIUM_PRETRAINED_MODELS_DICT # Avoiding circular imports with `space.py`

        name = GRAPHIUM_PRETRAINED_MODELS_DICT.get(name_or_path)

        if name is not None:
            return PredictorModule.load_from_checkpoint(
                GRAPHIUM_PRETRAINED_MODELS_DICT[name_or_path], map_location=device, strict=strict, **kwargs
            )

        if name is None and not (fs.exists(name_or_path) and fs.get_extension(name_or_path) == "ckpt"):
            raise ValueError(
                f"The model '{name_or_path}' is not available. Choose from {set(GRAPHIUM_PRETRAINED_MODELS_DICT.keys())} "
                "or pass a valid checkpoint (.ckpt) path."
            )

        return PredictorModule.load_from_checkpoint(name_or_path, map_location=device, strict=strict, **kwargs)

    def set_max_nodes_edges_per_graph(self, datamodule: BaseDataModule, stages: Optional[List[str]] = None):
        datamodule.setup()

        max_nodes = datamodule.get_max_num_nodes_datamodule(stages)
        max_edges = datamodule.get_max_num_edges_datamodule(stages)

        self.model.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)

    def get_num_graphs(self, data: Batch):
        """
        Method to compute number of graphs in a Batch.
        Essential to estimate throughput in graphs/s.
        """
        return torch.max(data.batch) + 1
