from goli.trainer.metrics import MetricWrapper
from typing import Dict, List, Any, Union, Any, Callable, Tuple, Type, Optional
import numpy as np
from copy import deepcopy
import time
from loguru import logger

import torch
from torch import nn, Tensor
import pytorch_lightning as pl
from torch_geometric.data import Data, Batch
from mup.optim import MuAdam

from goli.config.config_convert import recursive_config_reformating
from goli.trainer.predictor_options import EvalOptions, FlagOptions, ModelOptions, OptimOptions
from goli.trainer.predictor_summaries import TaskSummaries
from goli.data.datamodule import BaseDataModule
from goli.utils.moving_average_tracker import MovingAverageTracker

GOLI_PRETRAINED_MODELS = {
    "goli-zinc-micro-dummy-test": "gcs://goli-public/pretrained-models/goli-zinc-micro-dummy-test/model.ckpt"
}


class PredictorModule(pl.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        loss_fun: Dict[str, Union[str, Callable]],
        random_seed: int = 42,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        torch_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Optional[Union[str, int]] = None,
        multitask_handling: Optional[str] = None,
        metrics: Dict[str, Callable] = None,
        metrics_on_progress_bar: Dict[str, List[str]] = [],
        metrics_on_training_set: Optional[Dict[str, List[str]]] = None,
        flag_kwargs: Dict[str, Any] = None,
        task_norms: Optional[Dict[Callable, Any]] = None,
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
        """
        self.save_hyperparameters()

        self.random_seed = random_seed
        torch.random.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.target_nan_mask = target_nan_mask
        self.multitask_handling = multitask_handling
        self.task_norms = task_norms

        super().__init__()

        # Setting the model options
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
        # Setting the flag options
        self._flag_options = FlagOptions(flag_kwargs=flag_kwargs)

        self.model = self._model_options.model_class(**self._model_options.model_kwargs)
        self.tasks = list(loss_fun.keys())

        # Task-specific evalutation attributes
        self.loss_fun = {}
        self.metrics = {}
        self.metrics_on_progress_bar = {}
        self.metrics_on_training_set = {}
        for task in self.tasks:
            self.loss_fun[task] = EvalOptions.parse_loss_fun(loss_fun[task])
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

        # Initialize the epoch summary
        monitor = self.optim_options.scheduler_kwargs["monitor"].split("/")[0]
        mode = self.optim_options.scheduler_kwargs["mode"]

        self.task_epoch_summary = TaskSummaries(
            task_loss_fun=self.loss_fun,
            task_metrics=self.metrics,
            task_metrics_on_training_set=self.metrics_on_training_set,
            task_metrics_on_progress_bar=self.metrics_on_progress_bar,
            monitor=monitor,
            mode=mode,
        )

        # This helps avoid a bug when saving hparams to yaml with different dict or str formats
        self._set_hparams(recursive_config_reformating(self.hparams))

        # throughput estimation
        self.mean_val_time_tracker = MovingAverageTracker()
        self.mean_val_tput_tracker = MovingAverageTracker()
        self.epoch_start_time = None

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

    def configure_optimizers(self, impl=None):
        if impl is None:
            impl = torch.optim.Adam

        # Define the optimizer and schedulers
        optimiser = MuAdam(self.parameters(), **self.optim_options.optim_kwargs, impl=impl)
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
                metric=loss,
                threshold_kwargs=None,
                target_nan_mask=target_nan_mask,
                multitask_handling=multitask_handling,
            )
            for task, loss in loss_fun.items()
        }

        if weights is not None:
            raise NotImplementedError("Weights are no longer supported in the loss")
        all_task_losses = {
            task: wrapped(preds=preds[task], target=targets[task])
            for task, wrapped in wrapped_loss_fun_dict.items()
        }
        total_loss = torch.sum(torch.stack(list(all_task_losses.values())), dim=0)
        num_tasks = len(all_task_losses.keys())
        weighted_loss = total_loss / num_tasks
        return weighted_loss, all_task_losses

    def _general_step(self, batch: Dict[str, Tensor], step_name: str, to_cpu: bool) -> Dict[str, Any]:
        r"""Common code for training_step, validation_step and testing_step"""
        preds = self.forward(batch)  # The dictionary of predictions

        # * check for nan in model output
        targets_dict = batch.get("labels")

        # Different type of preds can be return by the forward
        if isinstance(preds, dict) and ("preds" in preds.keys()):
            preds = preds["preds"]
        elif isinstance(preds, Tensor):
            preds = {k: preds[ii] for ii, k in enumerate(targets_dict.keys())}

        # preds = {k: preds[ii] for ii, k in enumerate(targets_dict.keys())}
        for task, pred in preds.items():
            task_specific_norm = self.task_norms[task] if self.task_norms is not None else None
            # apply denormalization for predictions
            pred = task_specific_norm.denormalize(pred)
            if step_name == "train":
                # apply denormalization for targets
                targets_dict[task] = task_specific_norm.denormalize(targets_dict[task])
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

        device = "cpu" if to_cpu else None
        for task in preds:
            preds[task] = preds[task].detach().to(device=device)
            targets_dict[task] = targets_dict[task].detach().to(device=device)
        if weights is not None:
            weights = weights.detach().to(device=device)

        step_dict = {"preds": preds, "targets": targets_dict, "weights": weights}
        # step_dict[f"{self.loss_fun._get_name()}/{step_name}"] = loss.detach().cpu()            original

        # step_dict[f"weighted_loss/{step_name}"] = loss.detach().cpu()
        # step_dict[f"loss/{step_name}"] = loss.detach().cpu()
        for task in self.tasks:
            step_dict[
                self.task_epoch_summary.metric_log_name(task, self.loss_fun[task]._get_name(), step_name)
            ] = loss.detach()

        step_dict["loss"] = loss
        # print("loss ", self.global_step, self.current_epoch, loss)
        step_dict["task_losses"] = task_losses
        step_dict["gradient_norm"] = self.get_gradient_norm()
        return step_dict

    def flag_step(self, batch: Dict[str, Tensor], step_name: str, to_cpu: bool) -> Dict[str, Any]:
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

        device = "cpu" if to_cpu else None
        for key in preds.keys():
            preds[key] = preds[key].detach().to(device=device)
            targets[key] = targets[key].detach().to(device=device)
        if weights is not None:
            weights = weights.detach().to(device=device)

        step_dict = {"preds": preds, "targets": targets, "weights": weights}
        step_dict[f"loss/{step_name}"] = loss.detach().cpu()
        step_dict["loss"] = loss
        step_dict["task_losses"] = task_losses
        return step_dict

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        self.train_batch_start_time = time.time()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        train_batch_time = time.time() - self.train_batch_start_time
        num_graphs = self.get_num_graphs(batch["features"])
        tput = num_graphs / train_batch_time

        # this code is likely repeated for validation and testing, this should be moved to a function
        self.task_epoch_summary.update_predictor_state(
            step_name="train",
            targets=outputs["targets"],
            predictions=outputs["preds"],
            loss=outputs["loss"],  # This is the weighted loss for now, but change to task-specific loss
            task_losses=outputs["task_losses"],
            n_epochs=self.current_epoch,
        )
        metrics_logs = self.task_epoch_summary.get_metrics_logs()  # Dict[task, metric_logs]
        metrics_logs["_global"]["grad_norm"] = self.get_gradient_norm()
        outputs.update(metrics_logs)  # Dict[task, metric_logs]. Concatenate them?

        concatenated_metrics_logs = self.task_epoch_summary.concatenate_metrics_logs(metrics_logs)
        concatenated_metrics_logs["loss"] = outputs["loss"]
        outputs["grad_norm"] = self.get_gradient_norm()
        concatenated_metrics_logs["train/grad_norm"] = outputs["grad_norm"]
        concatenated_metrics_logs["train/batch_time"] = train_batch_time
        concatenated_metrics_logs["train/batch_tput"] = tput

        if self.logger is not None:
            self.logger.log_metrics(
                concatenated_metrics_logs, step=self.global_step
            )  # This is a pytorch lightning function call

    def training_step(self, batch: Dict[str, Tensor], to_cpu: bool = True) -> Dict[str, Any]:
        step_dict = None

        # Train using FLAG
        if self.flag_kwargs["n_steps"] > 0:
            step_dict = self.flag_step(batch=batch, step_name="train", to_cpu=to_cpu)
        # Train normally, without using FLAG
        elif self.flag_kwargs["n_steps"] == 0:
            # step_dict = self._general_step(batch=batch, step_name="train", to_cpu=True)
            step_dict = self._general_step(batch=batch, step_name="train", to_cpu=to_cpu)

        return step_dict  # Returning the metrics_logs with the loss

    def get_gradient_norm(self):
        # compute the norm
        total_norm = torch.tensor(0.0)
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        return total_norm

    def validation_step(self, batch: Dict[str, Tensor], to_cpu: bool = True) -> Dict[str, Any]:
        return self._general_step(batch=batch, step_name="val", to_cpu=to_cpu)

    def test_step(self, batch: Dict[str, Tensor], to_cpu: bool = True) -> Dict[str, Any]:
        return self._general_step(batch=batch, step_name="test", to_cpu=to_cpu)

    def _general_epoch_end(self, outputs: Dict[str, Any], step_name: str) -> None:
        r"""Common code for training_epoch_end, validation_epoch_end and testing_epoch_end"""
        # Transform the list of dict of dict, into a dict of list of dict
        preds = {}
        targets = {}
        device = device = outputs[0]["preds"][self.tasks[0]].device  # should be better way to do this
        # device = 0
        for task in self.tasks:
            preds[task] = torch.cat([out["preds"][task].to(device=device) for out in outputs], dim=0)
            targets[task] = torch.cat([out["targets"][task].to(device=device) for out in outputs], dim=0)
        if ("weights" in outputs[0].keys()) and (outputs[0]["weights"] is not None):
            weights = torch.cat([out["weights"] for out in outputs], dim=0)
        else:
            weights = None
        loss, task_losses = self.compute_loss(
            preds=preds,
            targets=targets,
            weights=weights,
            target_nan_mask=self.target_nan_mask,
            multitask_handling=self.multitask_handling,
            loss_fun=self.loss_fun,
        )

        self.task_epoch_summary.update_predictor_state(
            step_name=step_name,
            predictions=preds,
            targets=targets,
            loss=loss,
            task_losses=task_losses,
            n_epochs=self.current_epoch,
        )
        metrics_logs = self.task_epoch_summary.get_metrics_logs()
        self.task_epoch_summary.set_results(task_metrics=metrics_logs)

        return metrics_logs  # Consider returning concatenated dict for tensorboard

    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        if self.epoch_start_time is None:
            logger.warning("epoch timer not initialized")
        else:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_start_time = None
            self.log("epoch_time", torch.tensor(epoch_time))

    def training_epoch_end(self, outputs: Dict):
        """
        Nothing happens at the end of the training epoch.
        It serves no purpose to do a general step for the training,
        but it can explode the RAM when using a large dataset.
        """
        pass

    def on_validation_epoch_start(self) -> None:
        self.mean_val_time_tracker.reset()
        self.mean_val_tput_tracker.reset()
        return super().on_validation_epoch_start()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.validation_batch_start_time = time.time()
        return super().on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(self, outputs, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        val_batch_time = time.time() - self.validation_batch_start_time
        self.mean_val_time_tracker.update(val_batch_time)
        num_graphs = self.get_num_graphs(batch["features"])
        self.mean_val_tput_tracker.update(num_graphs / val_batch_time)
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def validation_epoch_end(self, outputs: Dict[str, Any]):
        metrics_logs = self._general_epoch_end(outputs=outputs, step_name="val")
        concatenated_metrics_logs = self.task_epoch_summary.concatenate_metrics_logs(metrics_logs)
        concatenated_metrics_logs["val/mean_time"] = self.mean_val_time_tracker.mean_value
        concatenated_metrics_logs["val/mean_tput"] = self.mean_val_tput_tracker.mean_value

        lr = self.optimizers().param_groups[0]["lr"]
        concatenated_metrics_logs["lr"] = lr
        concatenated_metrics_logs["n_epochs"] = self.current_epoch
        self.log_dict(concatenated_metrics_logs)

        # Save yaml file with the per-task metrics summaries
        full_dict = {}
        full_dict.update(self.task_epoch_summary.get_dict_summary())

    def test_epoch_end(self, outputs: Dict[str, Any]):
        metrics_logs = self._general_epoch_end(outputs=outputs, step_name="test")
        concatenated_metrics_logs = self.task_epoch_summary.concatenate_metrics_logs(metrics_logs)

        self.log_dict(concatenated_metrics_logs)

        # Save yaml file with the per-task metrics summaries
        full_dict = {}
        full_dict.update(self.task_epoch_summary.get_dict_summary())

    def on_train_start(self):
        hparams_log = deepcopy(self.hparams)
        hparams_log["n_params"] = self.n_params
        if self.logger is not None:
            self.logger.log_hyperparams(hparams_log)

    def get_progress_bar_dict(self) -> Dict[str, float]:
        prog_dict = {}
        prog_dict["loss"] = self.task_epoch_summary.weighted_loss.item()
        results_on_progress_bar = self.task_epoch_summary.get_results_on_progress_bar("val")
        for task in self.tasks:
            prog_dict[self.task_epoch_summary.metric_log_name(task, "loss", "val")] = (
                self.task_epoch_summary.task_summaries[task].summaries["val"].loss
            )
            prog_dict.update(results_on_progress_bar)
        return prog_dict

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
        return GOLI_PRETRAINED_MODELS

    @staticmethod
    def load_pretrained_models(name: str):
        """Load a pretrained model from its name.

        Args:
            name: Name of the model to load. List available
                from `goli.trainer.PredictorModule.list_pretrained_models()`.
        """

        if name not in GOLI_PRETRAINED_MODELS:
            raise ValueError(
                f"The model '{name}' is not available. Choose from {set(GOLI_PRETRAINED_MODELS.keys())}."
            )

        return PredictorModule.load_from_checkpoint(GOLI_PRETRAINED_MODELS[name])

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
