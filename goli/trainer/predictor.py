from goli.trainer.metrics import MetricWrapper
from typing import Dict, List, Any, Union, Any, Callable, Tuple, Type, Optional
import os
import numpy as np
from copy import deepcopy
import yaml
import dgl
from loguru import logger
from inspect import signature

import torch
from torch import nn, Tensor
import torch.optim.lr_scheduler as sc

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from goli.config.config_convert import recursive_config_reformating
from goli.utils.tensor import nan_mean, nan_std, nan_median
from goli.utils.fs import mkdir

LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "bce": torch.nn.BCELoss(),
    "l1": torch.nn.L1Loss(),
    "mae": torch.nn.L1Loss(),
}

GOLI_PRETRAINED_MODELS = {
    "goli-zinc-micro-dummy-test": "gcs://goli-public/pretrained-models/goli-zinc-micro-dummy-test/model.ckpt"
}

SCHEDULER_DICT = {
    "CosineAnnealingLR": sc.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": sc.CosineAnnealingWarmRestarts,
    "CyclicLR": sc.CyclicLR,
    "ExponentialLR": sc.ExponentialLR,
    "LambdaLR": sc.LambdaLR,
    "MultiStepLR": sc.MultiStepLR,
    "ReduceLROnPlateau": sc.ReduceLROnPlateau,
    "StepLR": sc.StepLR,
}


class EpochSummary:
    r"""Container for collecting epoch-wise results"""

    def __init__(self, monitor="loss", mode: str = "min", metrics_on_progress_bar=[]):
        self.monitor = monitor
        self.mode = mode
        self.metrics_on_progress_bar = metrics_on_progress_bar
        self.summaries = {}
        self.best_summaries = {}

    class Results:
        def __init__(
            self,
            targets: Tensor,
            predictions: Tensor,
            loss: float,
            metrics: dict,
            monitored_metric: str,
            n_epochs: int,
        ):
            self.targets = targets.detach().cpu()
            self.predictions = predictions.detach().cpu()
            self.loss = loss.item() if isinstance(loss, Tensor) else loss
            self.monitored_metric = monitored_metric
            if monitored_metric in metrics.keys():
                self.monitored = metrics[monitored_metric].detach().cpu()
            self.metrics = {key: value.tolist() for key, value in metrics.items()}
            self.n_epochs = n_epochs

    def set_results(self, name, targets, predictions, loss, metrics, n_epochs) -> float:
        metrics[f"loss/{name}"] = loss
        self.summaries[name] = EpochSummary.Results(
            targets=targets,
            predictions=predictions,
            loss=loss,
            metrics=metrics,
            monitored_metric=f"{self.monitor}/{name}",
            n_epochs=n_epochs,
        )
        if self.is_best_epoch(name, loss, metrics):
            self.best_summaries[name] = self.summaries[name]

    def is_best_epoch(self, name, loss, metrics):
        if not (name in self.best_summaries.keys()):
            return True

        metrics[f"loss/{name}"] = loss
        monitor_name = f"{self.monitor}/{name}"
        if not monitor_name in self.best_summaries.keys():
            return True

        if self.mode == "max":
            return metrics[monitor_name] > self.best_summaries[name].monitored
        elif self.mode == "min":
            return metrics[monitor_name] < self.best_summaries[name].monitored
        else:
            ValueError(f"Mode must be 'min' or 'max', provided `{self.mode}`")

    def get_results(self, name):
        return self.summaries[name]

    def get_best_results(self, name):
        return self.best_summaries[name]

    def get_results_on_progress_bar(self, name):
        results = self.summaries[name]
        results_prog = {
            f"{kk}/{name}": results.metrics[f"{kk}/{name}"] for kk in self.metrics_on_progress_bar
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


class PredictorModule(pl.LightningModule):
    def __init__(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Dict[str, Any],
        loss_fun: Union[str, Callable],
        random_seed: int = 42,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        lr_reduce_on_plateau_kwargs: Optional[Dict[str, Any]] = None,
        torch_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Optional[Union[int, float, str]] = None,
        metrics: Dict[str, Callable] = None,
        metrics_on_progress_bar: List[str] = [],
        metrics_on_training_set: Optional[List[str]] = None,
        flag_n_steps: int = 0,
        flag_alpha: float = 0.01,
    ):
        r"""
        A class that allows to use regression or classification models easily
        with Pytorch-Lightning.

        Parameters:
            model_class:
                pytorch module used to create a model

            model_kwargs:
                Key-word arguments used to initialize the model from `model_class`.

            loss_fun:
                Loss function used during training.
                Acceptable strings are 'mse', 'bce', 'mae', 'cosine'.
                Otherwise, a callable object must be provided, with a method `loss_fun._get_name()`.

            random_seed:
                The random seed used by Pytorch to initialize random tensors.

            optim_kwargs:
                Dictionnary used to initialize the optimizer, with possible keys below.

                - lr `float`: Learning rate (Default=`1e-3`)
                - weight_decay `float`: Weight decay used to regularize the optimizer (Default=`0.`)

            lr_reduce_on_plateau_kwargs:
                DEPRECIATED. USE `torch_scheduler_kwargs` instead.
                Dictionnary for the reduction of learning rate when reaching plateau, with possible keys below.

                - factor `float`: Factor by which to reduce the learning rate (Default=`0.5`)
                - patience `int`: Number of epochs without improvement to wait before reducing
                  the learning rate (Default=`10`)
                - mode `str`: One of min, max. In min mode, lr will be reduced when the quantity
                  monitored has stopped decreasing; in max mode it will be reduced when the quantity
                  monitored has stopped increasing. (Default=`"min"`).
                - min_lr `float`: A scalar or a list of scalars. A lower bound on the learning rate
                  of all param groups or each group respectively (Default=`1e-4`)

            torch_scheduler_kwargs:
                Dictionnary for the scheduling of learning rate, with possible keys below.

                - type `str`: Type of the learning rate to use from pytorch. Examples are
                  `'ReduceLROnPlateau'` (default), `'CosineAnnealingWarmRestarts'`, `'StepLR'`, etc.
                - **kwargs: Any other argument for the learning rate scheduler

            scheduler_kwargs:
                Dictionnary for the scheduling of the learning rate modification used by pytorch-lightning

                - monitor `str`: metric to track (Default=`"loss/val"`)
                - interval `str`: Whether to look at iterations or epochs (Default=`"epoch"`)
                - strict `bool`: if set to True will enforce that value specified in monitor is available
                  while trying to call scheduler.step(), and stop training if not found. If False will
                  only give a warning and continue training (without calling the scheduler). (Default=`True`)
                - frequency `int`: **TODO: NOT REALLY SURE HOW IT WORKS!** (Default=`1`)

            target_nan_mask:
                TODO: It's not implemented for the metrics yet!!

                - None: Do not change behaviour if there are nans

                - int, float: Value used to replace nans. For example, if `target_nan_mask==0`, then
                  all nans will be replaced by zeros

                - 'ignore': Nans will be ignored when computing the loss.

            metrics:
                A dictionnary of metrics to compute on the prediction, other than the loss function.
                These metrics will be logged into TensorBoard.

            metrics_on_progress_bar:
                The metrics names from `metrics` to display also on the progress bar of the training

            metrics_on_training_set:
                The metrics names from `metrics` to be computed on the training set for each iteration.
                If `None`, all the metrics are computed. Using less metrics can significantly improve
                performance, depending on the number of readouts.

            flag_n_steps:
                An integer that specifies the number of ascent steps when running FLAG during training.
                Default value of 0 trains GNNs without FLAG, and any value greater than 0 will use FLAG with that
                many iterations.

            flag_alpha:
                A float that specifies the ascent step size when running FLAG.
        """

        self.save_hyperparameters()

        torch.random.manual_seed(random_seed)
        np.random.seed(random_seed)

        super().__init__()
        self.model = model_class(**model_kwargs)

        # Basic attributes
        self.loss_fun = self.parse_loss_fun(loss_fun)
        self.random_seed = random_seed
        self.target_nan_mask = target_nan_mask
        self.metrics = metrics if metrics is not None else {}
        self.metrics_on_progress_bar = metrics_on_progress_bar
        self.metrics_on_training_set = (
            list(self.metrics.keys()) if metrics_on_training_set is None else metrics_on_training_set
        )
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.optim_kwargs = optim_kwargs
        self.flag_n_steps = flag_n_steps
        self.flag_alpha = flag_alpha

        # Set the default value for the optimizer
        self.optim_kwargs = optim_kwargs if optim_kwargs is not None else {}
        self.optim_kwargs.setdefault("lr", 1e-3)
        self.optim_kwargs.setdefault("weight_decay", 0.0)

        # Set the lightning scheduler
        self.scheduler_kwargs = {
                "interval": "epoch",
                "monitor": "loss/val",
                "mode": "min",
                "frequency": 1,
                "strict": True,}
        self.scheduler_kwargs.update(scheduler_kwargs)

        # Set the depreciated arguments, if provided
        if lr_reduce_on_plateau_kwargs is not None:
            logger.warning("`lr_reduce_on_plateau_kwargs` is depreciated, use `torch_scheduler_kwargs` instead.")
            if torch_scheduler_kwargs is not None:
                raise ValueError("Ambiguous. Both `lr_reduce_on_plateau_kwargs` and `torch_scheduler_kwargs` are provided")
            torch_scheduler_kwargs = {
                "type": "ReduceLROnPlateau",
                **lr_reduce_on_plateau_kwargs,
            }

        # Set the pytorch scheduler arguments
        if torch_scheduler_kwargs is None:
            self.torch_scheduler_kwargs = {}
        else:
            self.torch_scheduler_kwargs = torch_scheduler_kwargs
        self.torch_scheduler_kwargs.setdefault("type", "ReduceLROnPlateau")

        # Initialize the epoch summary
        monitor = self.scheduler_kwargs["monitor"].split("/")[0]
        mode = self.scheduler_kwargs["mode"]
        self.epoch_summary = EpochSummary(
            monitor, mode=mode, metrics_on_progress_bar=self.metrics_on_progress_bar
        )

        # This helps avoid a bug when saving hparams to yaml with different
        # dict or str formats
        self._set_hparams(recursive_config_reformating(self.hparams))

    @staticmethod
    def parse_loss_fun(loss_fun: Union[str, Callable]) -> Callable:
        r"""
        Parse the loss function from a string

        Parameters:
            loss_fun:
                A callable corresponding to the loss function or a string
                specifying the loss function from `LOSS_DICT`. Accepted strings are:
                "mse", "bce", "l1", "mae", "cosine".

        Returns:
            Callable:
                Function or callable to compute the loss, takes `preds` and `targets` as inputs.
        """

        if isinstance(loss_fun, str):
            loss_fun = LOSS_DICT[loss_fun]
        elif not callable(loss_fun):
            raise ValueError(f"`loss_fun` must be `str` or `callable`. Provided: {type(loss_fun)}")

        return loss_fun

    def forward(self, inputs: Dict) -> Dict[str, Any]:
        r"""
        Returns the result of `self.model.forward(*inputs)` on the inputs.
        """
        feats = self._convert_features_dtype(inputs["features"])
        out = {}
        out["preds"] = self.model.forward(feats)
        return out

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
        # Configure the parameters for the schedulers
        sc_kwargs = deepcopy(self.torch_scheduler_kwargs)
        scheduler_class = SCHEDULER_DICT[sc_kwargs.pop("type")]
        sig =  signature(scheduler_class.__init__)
        key_args = [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY]
        if ("monitor" in key_args):
            sc_kwargs.setdefault("monitor", self.scheduler_kwargs["monitor"])
        if ("mode" in key_args):
            sc_kwargs.setdefault("mode", self.scheduler_kwargs["mode"])

        # Define the optimizer and schedulers
        optimiser = torch.optim.Adam(self.parameters(), **self.optim_kwargs)
        torch_scheduler = scheduler_class(optimizer=optimiser, **sc_kwargs)
        scheduler = {
            "scheduler": torch_scheduler,
            **self.scheduler_kwargs,
        }
        return [optimiser], [scheduler]

    @staticmethod
    def compute_loss(
        preds: Tensor,
        targets: Tensor,
        weights: Optional[Tensor],
        loss_fun: Callable,
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

        wrapped_loss_fun = MetricWrapper(
            metric=loss_fun, threshold_kwargs=None, target_nan_mask=target_nan_mask
        )
        if weights is not None:
            raise NotImplementedError("Weights are no longer supported in the loss")
        loss = wrapped_loss_fun(preds=preds, target=targets)

        return loss

    def get_metrics_logs(
        self, preds: Tensor, targets: Tensor, weights: Optional[Tensor], step_name: str, loss: Tensor
    ) -> Dict[str, Any]:
        r"""
        Get the logs for the loss and the different metrics, in a format compatible with
        Pytorch-Lightning.

        Parameters:
            preds:
                Predicted values

            targets:
                Target values

            step_name:
                A string to mention whether the metric is computed on the training,
                validation or test set.

                - "train": On the training set
                - "val": On the validation set
                - "test": On the test set

        """

        targets = targets.to(dtype=preds.dtype, device=preds.device)

        # Compute the metrics always used in regression tasks
        metric_logs = {}
        metric_logs[f"mean_pred/{step_name}"] = nan_mean(preds)
        metric_logs[f"std_pred/{step_name}"] = nan_std(preds)
        metric_logs[f"median_pred/{step_name}"] = nan_median(preds)
        metric_logs[f"mean_target/{step_name}"] = nan_mean(targets)
        metric_logs[f"std_target/{step_name}"] = nan_std(targets)
        metric_logs[f"median_target/{step_name}"] = nan_median(targets)
        if torch.cuda.is_available():
            metric_logs[f"gpu_allocated_GB"] = torch.tensor(torch.cuda.memory_allocated() / (2 ** 30))

        # Specify which metrics to use
        metrics_to_use = self.metrics
        if step_name == "train":
            metrics_to_use = {
                key: metric for key, metric in metrics_to_use.items() if key in self.metrics_on_training_set
            }

        # Compute the additional metrics
        for key, metric in metrics_to_use.items():
            metric_name = f"{key}/{step_name}"
            try:
                metric_logs[metric_name] = metric(preds, targets)
            except Exception as e:
                metric_logs[metric_name] = torch.as_tensor(float("nan"))

        # Convert all metrics to CPU, except for the loss
        metric_logs[f"{self.loss_fun._get_name()}/{step_name}"] = loss.detach().cpu()
        metric_logs = {key: metric.detach().cpu() for key, metric in metric_logs.items()}

        return metric_logs

    def _general_step(
        self, batch: Dict[str, Tensor], batch_idx: int, step_name: str, to_cpu: bool
    ) -> Dict[str, Any]:
        r"""Common code for training_step, validation_step and testing_step"""
        preds = self.forward(batch)["preds"]
        targets = batch.pop("labels").to(dtype=preds.dtype)
        weights = batch.pop("weights", None)

        loss = self.compute_loss(
            preds=preds,
            targets=targets,
            weights=weights,
            target_nan_mask=self.target_nan_mask,
            loss_fun=self.loss_fun,
        )

        device = "cpu" if to_cpu else None
        preds = preds.detach().to(device=device)
        targets = targets.detach().to(device=device)
        if weights is not None:
            weights = weights.detach().to(device=device)

        step_dict = {"preds": preds, "targets": targets, "weights": weights}
        step_dict[f"{self.loss_fun._get_name()}/{step_name}"] = loss.detach().cpu()
        return loss, step_dict

    def flag_step(
        self, batch: Dict[str, Tensor], batch_idx: int, step_name: str, to_cpu: bool
    ) -> Dict[str, Any]:
        r"""
        Perform adversarial data agumentation during one training step using FLAG.
        Paper: https://arxiv.org/abs/2010.09891
        Github: https://github.com/devnkong/FLAG
        """

        X = self._convert_features_dtype(batch["features"])
        X_shape = X.ndata["feat"].shape

        pert = torch.FloatTensor(X_shape).uniform_(-self.flag_alpha, self.flag_alpha).to(device=X.device)
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
            / self.flag_n_steps
        )

        # Iteratively augment data by applying perturbations
        # Accumulate the gradients to be applied to the weights of the network later on
        for _ in range(self.flag_n_steps - 1):
            loss.backward()
            pert_data = pert.detach() + self.flag_alpha * torch.sign(pert.grad.detach())
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
                / self.flag_n_steps
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
        if self.flag_n_steps > 0:
            loss, step_dict = self.flag_step(batch=batch, batch_idx=batch_idx, step_name="train", to_cpu=True)
        # Train normally, without using FLAG
        elif self.flag_n_steps == 0:
            loss, step_dict = self._general_step(
                batch=batch, batch_idx=batch_idx, step_name="train", to_cpu=True
            )

        metrics_logs = self.get_metrics_logs(
            preds=step_dict["preds"],
            targets=step_dict["targets"],
            weights=step_dict["weights"],
            step_name="train",
            loss=loss,
        )

        step_dict.update(metrics_logs)
        step_dict["loss"] = loss

        self.logger.log_metrics(metrics_logs, step=self.global_step)

        # Predictions and targets are no longer needed after the step.
        # Keeping them will increase memory usage significantly for large datasets.
        step_dict.pop("preds")
        step_dict.pop("targets")
        step_dict.pop("weights")

        return step_dict

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        return self._general_step(batch=batch, batch_idx=batch_idx, step_name="val", to_cpu=True)[1]

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Any]:
        return self._general_step(batch=batch, batch_idx=batch_idx, step_name="val", to_cpu=True)[1]

    def _general_epoch_end(self, outputs: Dict[str, Any], step_name: str) -> None:
        r"""Common code for training_epoch_end, validation_epoch_end and testing_epoch_end"""

        # Transform the list of dict of dict, into a dict of list of dict
        preds = torch.cat([out["preds"] for out in outputs], dim=0)
        targets = torch.cat([out["targets"] for out in outputs], dim=0)
        if outputs[0]["weights"] is not None:
            weights = torch.cat([out["weights"] for out in outputs], dim=0)
        else:
            weights = None
        loss = self.compute_loss(
            preds=preds,
            targets=targets,
            weights=weights,
            target_nan_mask=self.target_nan_mask,
            loss_fun=self.loss_fun,
        )
        metrics_logs = self.get_metrics_logs(
            preds=preds, targets=targets, weights=weights, step_name=step_name, loss=loss
        )

        self.epoch_summary.set_results(
            name=step_name,
            predictions=preds,
            targets=targets,
            loss=loss,
            metrics=metrics_logs,
            n_epochs=self.current_epoch,
        )

        return metrics_logs

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

        # Save yaml file with the metrics summaries
        full_dict = {}
        full_dict.update(self.epoch_summary.get_dict_summary())
        tb_path = self.logger.log_dir

        # Write the YAML file with the metrics
        if self.current_epoch >= 1:
            mkdir(tb_path)
            with open(os.path.join(tb_path, "metrics.yaml"), "w") as file:
                yaml.dump(full_dict, file)

    def test_epoch_end(self, outputs: Dict[str, Any]):

        metrics_logs = self._general_epoch_end(outputs=outputs, step_name="test")
        self.log_dict(metrics_logs)

        # Save yaml file with the metrics summaries
        full_dict = {}
        full_dict.update(self.epoch_summary.get_dict_summary())
        tb_path = self.logger.log_dir
        os.makedirs(tb_path, exist_ok=True)
        with open(f"{tb_path}/metrics.yaml", "w") as file:
            yaml.dump(full_dict, file)

    def on_train_start(self):
        hparams_log = deepcopy(self.hparams)
        hparams_log["n_params"] = self.n_params
        self.logger.log_hyperparams(hparams_log, self.epoch_summary.get_results("val").metrics)

    def get_progress_bar_dict(self) -> Dict[str, float]:
        prog_dict = super().get_progress_bar_dict()
        results_on_progress_bar = self.epoch_summary.get_results_on_progress_bar("val")
        prog_dict["loss/val"] = self.epoch_summary.summaries["val"].loss
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
