from goli.trainer.metrics import MetricWrapper
from typing import Dict, List, Any, Union, Any, Callable, Tuple, Type, Optional
import os
import numpy as np
from copy import deepcopy
import yaml
import dgl
from loguru import logger
from inspect import signature
from pprint import pprint

import torch
from torch import nn, Tensor

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from goli.config.config_convert import recursive_config_reformating
from goli.trainer.predictor_options import EvalOptions, FlagOptions, ModelOptions, OptimOptions
from goli.trainer.predictor_summaries import Summary, TaskSummaries
from goli.utils.fs import mkdir
from goli.utils.spaces import SCHEDULER_DICT

GOLI_PRETRAINED_MODELS = {
    "goli-zinc-micro-dummy-test": "gcs://goli-public/pretrained-models/goli-zinc-micro-dummy-test/model.ckpt"
}


class PredictorModule(pl.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],                                   # Leave
        model_kwargs: Dict[str, Any],                                   # Leave
        loss_fun: Dict[str, Union[str, Callable]],                      # Task-specific
        random_seed: int = 42,                                          # Leave
        optim_kwargs: Optional[Dict[str, Any]] = None,                  # Leave for now
        lr_reduce_on_plateau_kwargs: Optional[Dict[str, Any]] = None,   # Leave for now
        torch_scheduler_kwargs: Optional[Dict[str, Any]] = None,        # Leave for now
        scheduler_kwargs: Optional[Dict[str, Any]] = None,              # Leave for now
        target_nan_mask: Optional[Union[int, float, str]] = None,       # Leave
        metrics: Dict[str, Callable] = None,                            # Task-specific
        metrics_on_progress_bar: List[str] = [],                        # Task-specific
        metrics_on_training_set: Optional[List[str]] = None,            # Task-specific
        flag_kwargs: Dict[str, Any] = None,
    ):
        self.save_hyperparameters()

        self.random_seed = random_seed
        torch.random.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.target_nan_mask = target_nan_mask

        super().__init__()

        model_options = ModelOptions(
            model_class=model_class,
            model_kwargs=model_kwargs
        )
        optim_options = OptimOptions(
            optim_kwargs=optim_kwargs,
            lr_reduce_on_plateau_kwargs=lr_reduce_on_plateau_kwargs,
            torch_scheduler_kwargs=torch_scheduler_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )
        #eval_options = EvalOptions(
        #    loss_fun=loss_fun,
        #    metrics=metrics,
        #    metrics_on_progress_bar=metrics_on_progress_bar,
        #    metrics_on_training_set=metrics_on_training_set
        #)
        # Assume task-specific eval options
        eval_options = {}
        for task in loss_fun:
            eval_options[task] = EvalOptions(
                loss_fun=loss_fun[task],
                metrics=metrics[task],
                metrics_on_progress_bar=metrics_on_progress_bar[task],
                metrics_on_training_set=metrics_on_training_set[task] if metrics_on_training_set is not None else None
            )
        flag_options = FlagOptions(
            flag_kwargs=flag_kwargs
        )

        self._model_options = model_options
        self._optim_options = optim_options
        self._eval_options_dict: Dict[str, EvalOptions] = eval_options
        self._flag_options = flag_options

        self.model = self._model_options.model_class(**self._model_options.model_kwargs)
        self.tasks = list(loss_fun.keys())

        # Basic attributes
###########################################################################################################################################
        #self.loss_fun = {"micro_zinc": EvalOptions.parse_loss_fun(loss_fun["micro_zinc"])}

        #self.loss_fun = self._eval_options.parse_loss_fun(self._eval_options.loss_fun)              # Look into this
        #self.metrics = self._eval_options.metrics if self._eval_options.metrics is not None else {}
        #self.metrics_on_progress_bar = self._eval_options.metrics_on_progress_bar
        #self.metrics_on_training_set = (
        #    list(self.metrics.keys()) if self._eval_options.metrics_on_training_set is None else self._eval_options.metrics_on_training_set
        #)

        # task-specific evalutation attributes
        self.loss_fun = {}
        self.metrics = {}
        self.metrics_on_progress_bar = {}
        self.metrics_on_training_set = {}
        for task in self.tasks:
            self.loss_fun[task] = EvalOptions.parse_loss_fun(loss_fun[task])
            self.metrics[task] = self._eval_options_dict[task].metrics if self._eval_options_dict[task].metrics is not None else {}
            self.metrics_on_progress_bar[task] = self._eval_options_dict[task].metrics_on_progress_bar
            self.metrics_on_training_set[task] = (
                list(self.metrics[task].keys()) if self._eval_options_dict[task].metrics_on_training_set is None else self._eval_options_dict[task].metrics_on_training_set
            )
###########################################################################################################################################
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Set the parameters and default values for the FLAG adversarial augmentation, and check values
        self._flag_options.set_kwargs()
        self.flag_kwargs = self._flag_options.flag_kwargs

        # Set the parameters for optimizer options
        self._optim_options.set_kwargs()
        # Set the parameters and default value for the optimizer, and check values
        self.optim_kwargs = self._optim_options.optim_kwargs
        # Set the lightning scheduler
        self.scheduler_kwargs = self._optim_options.scheduler_kwargs
        # Set the pytorch scheduler arguments
        self.torch_scheduler_kwargs = self._optim_options.torch_scheduler_kwargs

###########################################################################################################################################
        # Initialize the epoch summary
        monitor = "micro_zinc/MSELoss/val" #self.scheduler_kwargs["monitor"].split("/")[0] TODO: Fix the scheduler with the Summary class
        mode = "min" #self.scheduler_kwargs["mode"]
        #self.epoch_summary = Summary(
        #    loss_fun=self.loss_fun,
        #    metrics=self.metrics,
        #    metrics_on_training_set=self.metrics_on_training_set,
        #    metrics_on_progress_bar=self.metrics_on_progress_bar,
        #    monitor=monitor,
        #    mode=mode,
        #)

        self.task_epoch_summary = TaskSummaries(
            task_loss_fun=self.loss_fun,
            task_metrics=self.metrics,
            task_metrics_on_training_set=self.metrics_on_training_set,
            task_metrics_on_progress_bar=self.metrics_on_progress_bar,
            monitor=monitor,
            mode=mode,
        )
###########################################################################################################################################

        # This helps avoid a bug when saving hparams to yaml with different dict or str formats
        self._set_hparams(recursive_config_reformating(self.hparams))

    def forward(self, inputs: Dict) -> Dict[str, Union[torch.Tensor, Any]]:
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
        out = self.model.forward(feats)

        # Convert the output of the model to a dictionary
        if isinstance(out, dict) and ("preds" in out.keys()):
            out_dict = out
        else:
            out_dict = {"preds": out}

        return out_dict

    def _convert_features_dtype(self, feats):
        # Convert features to dtype
        if isinstance(feats, torch.Tensor):
            feats = feats.to(self.dtype)
        elif isinstance(feats, dgl.DGLHeteroGraph):
            for key, val in feats.ndata.items():
                if isinstance(val, torch.Tensor):
                    feats.ndata[key] = val.to(dtype=self.dtype)
            for key, val in feats.edata.items():
                if isinstance(val, torch.Tensor):
                    feats.edata[key] = val.to(dtype=self.dtype)

        return feats

    def configure_optimizers(self):
        # TODO: Fix scheduling with the Summary class
        # Configure the parameters for the schedulers
        # sc_kwargs = deepcopy(self.torch_scheduler_kwargs)
        # scheduler_class = SCHEDULER_DICT[sc_kwargs.pop("module_type")]
        # sig = signature(scheduler_class.__init__)
        # key_args = [p.name for p in sig.parameters.values()]
        # if "monitor" in key_args:
        #     sc_kwargs.setdefault("monitor", self.scheduler_kwargs["monitor"])
        # if "mode" in key_args:
        #     sc_kwargs.setdefault("mode", self.scheduler_kwargs["mode"])

        # Define the optimizer and schedulers
        optimiser = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        # torch_scheduler = scheduler_class(optimizer=optimiser, **sc_kwargs)
        # scheduler = {
        #     "scheduler": torch_scheduler,
        #     **self.scheduler_kwargs,
        # }
        #scheduler = None
        #return [optimiser], [scheduler]
        return [optimiser]

    @staticmethod
    def compute_loss(
        preds: Dict[str, Tensor],
        targets: Dict[str, Tensor],
        weights: Optional[Tensor],
        loss_fun: Dict[str, Callable],                  # Was Callable
        target_nan_mask: Union[Type, str] = "ignore",
    ) -> Tensor:
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

                - 'ignore-flatten': The Tensor will be reduced to a vector without the NaN values.

                - 'ignore-mean-label': NaNs will be ignored when computing the loss. Note that each column
                  has a different number of NaNs, so the metric will be computed separately
                  on each column, and the metric result will be averaged over all columns.
                  *This option might slowdown the computation if there are too many labels*

            loss_fun:
                Loss function to use

        Returns:
            Tensor:
                Resulting loss
        """
        #wrapped_loss_fun = MetricWrapper(metric=loss_fun, threshold_kwargs=None, target_nan_mask=target_nan_mask)
        wrapped_loss_fun_dict = {task: MetricWrapper(metric=loss, threshold_kwargs=None, target_nan_mask=target_nan_mask) for task, loss in loss_fun.items()}

        if weights is not None:
            raise NotImplementedError("Weights are no longer supported in the loss")
        #loss = wrapped_loss_fun(preds=preds, target=targets)
        all_task_losses = {task: wrapped(preds=preds[task], target=targets[task]) for task, wrapped in wrapped_loss_fun_dict.items()}
        #total_loss = torch.sum(all_task_losses.values(), dim=0)
        total_loss = torch.zeros_like(next(iter(all_task_losses.values())), requires_grad=True)
        for task, loss in all_task_losses.items():
            total_loss = torch.add(total_loss, loss)
        num_tasks = len(all_task_losses.keys())
        weighted_loss = torch.div(total_loss, num_tasks)
        return weighted_loss, all_task_losses            # Return all_task_losses?

    def _general_step(
        self, batch: Dict[str, Tensor], batch_idx: int, step_name: str, to_cpu: bool
    ) -> Dict[str, Any]:
        r"""Common code for training_step, validation_step and testing_step"""
        preds = self.forward(batch)["preds"]                    # The dictionary of predictions
        #targets = batch.pop("labels").to(dtype=preds.dtype)
        targets_dict = batch.pop("labels")
        for task, pred in preds.items():
            targets_dict[task] = targets_dict[task].to(dtype=pred.dtype)
        weights = batch.pop("weights", None)

        loss, task_losses = self.compute_loss(
            preds=preds,
            targets=targets_dict,
            weights=weights,
            target_nan_mask=self.target_nan_mask,
            loss_fun=self.loss_fun,                         # This is a dictionary
        )

        device = "cpu" if to_cpu else None
        #preds = preds.detach().to(device=device)
        #targets = targets.detach().to(device=device)
        for task in preds:
            preds[task] = preds[task].detach().to(device=device)
            targets_dict[task] = targets_dict[task].detach().to(device=device)
        if weights is not None:
            weights = weights.detach().to(device=device)

        step_dict = {"preds": preds, "targets": targets_dict, "weights": weights}
        #step_dict[f"{self.loss_fun._get_name()}/{step_name}"] = loss.detach().cpu()            original

        #step_dict[f"weighted_loss/{step_name}"] = loss.detach().cpu()
        #step_dict[f"loss/{step_name}"] = loss.detach().cpu()
        for task in self.tasks:     # TODO: Verify consistency with Summary class
            step_dict[self.task_epoch_summary.metric_log_name(task, self.loss_fun[task]._get_name(), step_name)] = loss.detach().cpu()

        step_dict["task_losses"] = task_losses
        return loss, step_dict

    def flag_step(
        self, batch: Dict[str, Tensor], batch_idx: int, step_name: str, to_cpu: bool
    ) -> Dict[str, Any]:
        r"""
        Perform adversarial data agumentation during one training step using FLAG.
        Paper: https://arxiv.org/abs/2010.09891
        Github: https://github.com/devnkong/FLAG
        """

        alpha, n_steps = self.flag_kwargs["alpha"], self.flag_kwargs["n_steps"]

        X = self._convert_features_dtype(batch["features"])
        X_shape = X.ndata["feat"].shape

        pert = torch.FloatTensor(X_shape).uniform_(-alpha, alpha).to(device=X.device)
        pert.requires_grad = True

        # Perturb the features
        pert_batch = deepcopy(batch)
        pert_batch["features"].ndata["feat"] = batch["features"].ndata["feat"] + pert

        preds = self.forward(pert_batch)["preds"]
        targets = batch.pop("labels").to(dtype=preds.dtype)
        weights = batch.pop("weights", None)
        loss = (
            self.compute_loss(
                preds=preds,
                targets=targets,
                weights=weights,
                target_nan_mask=self.target_nan_mask,
                loss_fun=self.loss_fun,
            )
            / n_steps
        )

        # Iteratively augment data by applying perturbations
        # Accumulate the gradients to be applied to the weights of the network later on
        for _ in range(n_steps - 1):
            loss.backward()
            pert_data = pert.detach() + alpha * torch.sign(pert.grad.detach())
            pert.data = pert_data.data
            pert.grad[:] = 0
            pert_batch["features"].ndata["feat"] = batch["features"].ndata["feat"] + pert
            preds = self.forward(pert_batch)["preds"]
            loss = (
                self.compute_loss(
                    preds=preds,
                    targets=targets,
                    weights=weights,
                    target_nan_mask=self.target_nan_mask,
                    loss_fun=self.loss_fun,
                )
                / n_steps
            )

        device = "cpu" if to_cpu else None
        preds = preds.detach().to(device=device)
        targets = targets.detach().to(device=device)
        if weights is not None:
            weights = weights.detach().to(device=device)

        step_dict = {"preds": preds, "targets": targets, "weights": weights}
        step_dict[f"{self.loss_fun._get_name()}/{step_name}"] = loss.detach().cpu()
        return loss, step_dict

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        loss = None
        step_dict = None

        # Train using FLAG
        if self.flag_kwargs["n_steps"] > 0:
            loss, step_dict = self.flag_step(batch=batch, batch_idx=batch_idx, step_name="train", to_cpu=True)
        # Train normally, without using FLAG
        elif self.flag_kwargs["n_steps"] == 0:
            loss, step_dict = self._general_step(
                batch=batch, batch_idx=batch_idx, step_name="train", to_cpu=True
            )

#################################################################################################################
        self.task_epoch_summary.update_predictor_state(
            step_name="train",
            targets=step_dict["targets"],
            predictions=step_dict["preds"],
            loss=loss,              # This is the weighted loss for now, but change to task-sepcific loss
            task_losses=step_dict["task_losses"],
            n_epochs=self.current_epoch,
        )
        metrics_logs = self.task_epoch_summary.get_metrics_logs()       # Dict[task, metric_logs]


        step_dict.update(metrics_logs)          # Dict[task, metric_logs]. Concatenate them?
        step_dict["loss"] = loss                # Update loss per task or weighted loss?
        step_dict.pop("task_losses")            # The task losses are already contained in metrics_logs

        self.logger.log_metrics(metrics_logs, step=self.global_step)            # This is a pytorch lightning function call
#################################################################################################################


        # Predictions and targets are no longer needed after the step.
        # Keeping them will increase memory usage significantly for large datasets.
        step_dict.pop("preds")
        step_dict.pop("targets")
        step_dict.pop("weights")

        return step_dict  # Returning the metrics_logs with the loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        # We don't need the task losses to be logged since it's taken care of in the summary
        step_dict = self._general_step(batch=batch, batch_idx=batch_idx, step_name="val", to_cpu=True)[1]
        step_dict.pop("task_losses")
        return step_dict

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        # We don't need the task losses to be logged since it's taken care of in the summary
        step_dict = self._general_step(batch=batch, batch_idx=batch_idx, step_name="test", to_cpu=True)[1]
        step_dict.pop("task_losses")
        return step_dict

    # So it's over the entire dataset
    def _general_epoch_end(self, outputs: Dict[str, Any], step_name: str) -> None:
        r"""Common code for training_epoch_end, validation_epoch_end and testing_epoch_end"""
        # epoch_end returns a list of all the output from the _step
        # Transform the list of dict of dict, into a dict of list of dict
        #preds = torch.cat([out["preds"] for out in outputs], dim=0)         # outputs is a list of the output
        #targets = torch.cat([out["targets"] for out in outputs], dim=0)
        preds = {}
        targets = {}
        for task in self.tasks:
            preds[task] = torch.cat([out["preds"][task] for out in outputs], dim=0)
            targets[task] = torch.cat([out["targets"][task] for out in outputs], dim=0)
        if outputs[0]["weights"] is not None:
            weights = torch.cat([out["weights"] for out in outputs], dim=0)
        else:
            weights = None
        loss, task_losses = self.compute_loss(
            preds=preds,
            targets=targets,
            weights=weights,
            target_nan_mask=self.target_nan_mask,
            loss_fun=self.loss_fun,
        )
################################################################################################################
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
################################################################################################################
        return metrics_logs             # Consider returning concatenated dict for tensorboard

    def training_epoch_end(self, outputs: Dict):
        """
        Nothing happens at the end of the training epoch.
        It serves no purpose to do a general step for the training,
        but it can explode the RAM when using a large dataset.
        """
        pass

    def validation_epoch_end(self, outputs: Dict[str, Any]):

        metrics_logs = self._general_epoch_end(outputs=outputs, step_name="val")

        lr = self.optimizers().param_groups[0]["lr"]
        metrics_logs["lr"] = lr
        metrics_logs["n_epochs"] = self.current_epoch
        self.log_dict(metrics_logs)

        # TODO: Save to YAML
        # Save yaml file with the metrics summaries
        #full_dict = {}
        #full_dict.update(self.epoch_summary.get_dict_summary())
        #tb_path = self.logger.log_dir

        # Write the YAML file with the metrics
        #if self.current_epoch >= 0:
        #    mkdir(tb_path)
        #    with open(os.path.join(tb_path, "metrics.yaml"), "w") as file:
        #        yaml.dump(full_dict, file)

    def test_epoch_end(self, outputs: Dict[str, Any]):

        metrics_logs = self._general_epoch_end(outputs=outputs, step_name="test")
        self.log_dict(metrics_logs)

        # TODO: Save to YAML
        # Save yaml file with the metrics summaries
        #full_dict = {}
        #full_dict.update(self.epoch_summary.get_dict_summary())
        #tb_path = self.logger.log_dir
        #os.makedirs(tb_path, exist_ok=True)
        #with open(f"{tb_path}/metrics.yaml", "w") as file:
        #    yaml.dump(full_dict, file)

    def on_train_start(self):
        hparams_log = deepcopy(self.hparams)
        hparams_log["n_params"] = self.n_params
        #self.logger.log_hyperparams(hparams_log, self.epoch_summary.get_results("val").metrics)

    def get_progress_bar_dict(self) -> Dict[str, float]:
#####################################################################################################################
        #prog_dict = super().get_progress_bar_dict()
        #results_on_progress_bar = self.epoch_summary.get_results_on_progress_bar("val")
        #prog_dict["loss/val"] = self.epoch_summary.summaries["val"].loss
        #prog_dict.update(results_on_progress_bar)

        # TODO: Make sure the keys match up.
        prog_dict = {task: {} for task in self.tasks}
        results_on_progress_bar = self.task_epoch_summary.get_results_on_progress_bar("val")
        for task in self.tasks:
            prog_dict[task][self.task_epoch_summary.metric_log_name(task, "loss", "val")] = self.task_epoch_summary.task_summaries[task].summaries["val"].loss
            prog_dict.update(results_on_progress_bar)
#####################################################################################################################
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
