from typing import Dict, List, Any, Union, Any, Callable, Tuple, Type, Optional
import os, math
import torch
import dgl
import numpy as np
import pytorch_lightning as pl
from copy import deepcopy

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from goli.trainer.reporting import ModelSummaryExtended


LOSS_DICT = {
    "mse": torch.nn.MSELoss(),
    "bce": torch.nn.BCELoss(),
    "l1": torch.nn.L1Loss(),
    "mae": torch.nn.L1Loss(),
    "cosine": torch.nn.CosineEmbeddingLoss(),
}


class EpochSummary:
    r"""Container for collecting epoch-wise results"""

    class Results:
        def __init__(self, targets: torch.Tensor, predictions: torch.Tensor, loss: float, metrics: dict):
            self.targets = targets
            self.predictions = predictions
            self.loss = loss
            self.metrics = {key: value.tolist() for key, value in metrics.items()}

    def __init__(self):
        self.summaries = {}

    def set_results(self, name, targets, predictions, loss, metrics) -> float:
        self.summaries[name] = EpochSummary.Results(targets, predictions, loss, metrics)

    def get_results(self, name):
        return self.summaries[name]


class PredictorModule(pl.LightningModule):
    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        loss_fun: Union[str, Callable],
        random_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
        weight_decay: float = 0.0,
        optim_kwargs: Optional[Dict[str, Any]] = None,
        lr_reduce_on_plateau_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        target_nan_mask: Union[int, float, str, type(None)] = None,
        metrics: Dict[str, Callable] = None,
        metrics_on_progress_bar: List[str] = [],
        collate_fn: Union[type(None), Callable] = None,
        additional_hparams: Dict[str, Any] = None,
    ):
        r"""
        A class that allows to use regression or classification models easily
        with Pytorch-Lightning.

        Parameters:
            model:
                Pytorch model trained on the classification/regression task

            loss_fun:
                Loss function used during training.
                Acceptable strings are 'mse', 'bce', 'mae', 'cosine'.
                Otherwise, a callable object must be provided, with a method `loss_fun._get_name()`.

            lr:
                The learning rate used during the training.

            random_seed:
                The random seed used by Pytorch to initialize random tensors.

            dtype:
                The desired floating point type of the floating point parameters and buffers in this module.

            device:
                the desired device of the parameters and buffers in this module

            weight_decay:
                Weight decay used to regularize the optimizer

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

            collate_fn:
                Merges a list of samples to form a mini-batch of Tensor(s).
                Used when using batched loading from a map-style dataset.
                See `torch.utils.data.DataLoader.__init__`

            additional_hparams:
                Additionnal hyper-parameters to log in the TensorBoard file.
                They won't be used by the class, only logged.

        """

        torch.random.manual_seed(random_seed)
        np.random.seed(random_seed)

        super().__init__()

        # Basic attributes
        self.model = model_class(**model_kwargs).to(dtype=dtype, device=device)
        self.loss_fun = self.parse_loss_fun(loss_fun)
        self.random_seed = random_seed
        self.target_nan_mask = target_nan_mask
        self.metrics = metrics if metrics is not None else {}
        self.metrics_on_progress_bar = metrics_on_progress_bar
        self.collate_fn = collate_fn
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.epoch_summary = EpochSummary()
        self._dtype = dtype
        self._device = device

        # Set the default value for the optimizer
        self.optim_kwargs = optim_kwargs if optim_kwargs is not None else {}
        self.optim_kwargs.set_default('lr', 1e-3)
        self.optim_kwargs.set_default('weight_decay', 0)

        self.lr_reduce_on_plateau_kwargs = lr_reduce_on_plateau_kwargs if lr_reduce_on_plateau_kwargs is not None else {}
        self.lr_reduce_on_plateau_kwargs.set_default('factor', 0.5)
        self.lr_reduce_on_plateau_kwargs.set_default('patience', 7)

        self.optim_kwargs = optim_kwargs if optim_kwargs is not None else {}
        self.scheduler_kwargs.set_default('monitor', 'val_loss')
        self.scheduler_kwargs.set_default('interval', 'epoch')
        self.scheduler_kwargs.set_default('frequency', 1)
        self.scheduler_kwargs.set_default('strict', True)

        # Gather the hyper-parameters of the model
        self.hparams = deepcopy(self.model.hparams) if hasattr(self.model, "hparams") else {}
        if additional_hparams is not None:
            self.hparams.update(additional_hparams)

        # Add other hyper-parameters to the list
        self.hparams.update(
            {
                "loss_fun": self.loss_fun._get_name(),
                "random_seed": self.random_seed,
                "n_params": self.n_params,
            }
        )

        self.hparams.update({f'optim.{key}': val for key, val in self.optim_kwargs.items()})
        self.hparams.update({f'lr_reduce.{key}': val for key, val in self.lr_reduce_on_plateau_kwargs.items()})

        self.to(dtype=dtype, device=device)

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

    def forward(self, *inputs: List[torch.Tensor]):
        r"""
        Returns the result of `self.model.forward(*inputs)` on the inputs.
        """
        out = self.model.forward(*inputs)
        return out

    def prepare_data(self):
        r"""
        Method that parses the training and validation datasets, and creates
        the samplers. The following attributes are set by the current method.

        Attributes:
            dataset (Dataset):
                Either the full dataset, or the training dataset, depending on
                if the validation is provided as a split percentage, or as
                a stand-alone dataset.
            val_dataset (Dataset):
                Either a stand-alone dataset used for validation, or a pointer
                copy of the `dataset`.
            train_sampler (SubsetRandomSampler):
                The sampler for the training set
            val_sampler (SubsetRandomSampler):
                The sampler for the validation set

        """

        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))

        if isinstance(self.val_split, float):
            split = int(np.floor(self.val_split * dataset_size))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            self.val_dataset = self.dataset

            # Creating data samplers and loaders:
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)

        elif isinstance(self.val_split, Dataset):
            train_indices = list(range(dataset_size))
            val_indices = list(range(len(self.val_split)))
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.val_sampler = SubsetRandomSampler(val_indices)
            self.val_dataset = self.val_split

        else:
            raise ValueError("Unsupported validation split")

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), **self.optim_kwargs)

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer=optimiser, **self.lr_reduce_on_plateau_kwargs
            ),
            **self.scheduler_kwargs,
        }
        return [optimiser], [scheduler]

    @staticmethod
    def compute_loss(
        preds: torch.Tensor,
        targets: torch.Tensor,
        loss_fun: Callable,
        target_nan_mask: Union[Type, str] = "ignore",
    ) -> torch.Tensor:
        r"""
        Compute the loss using the specified loss function, and dealing with
        the nans in the `targets`.

        Parameters:
            preds:
                Predicted values

            targets:
                Target values

            target_nan_mask:

                - None: Do not change behaviour if there are nans

                - int, float: Value used to replace nans. For example, if `target_nan_mask==0`, then
                  all nans will be replaced by zeros

                - 'ignore': Nans will be ignored when computing the loss.

            loss_fun:
                Loss function to use

        Returns:
            torch.Tensor:
                Resulting loss
        """
        if target_nan_mask is None:
            pass
        elif isinstance(target_nan_mask, (int, float)):
            targets[torch.isnan(targets)] = target_nan_mask
        elif target_nan_mask == "ignore":
            nans = torch.isnan(targets)
            targets = targets[~nans]
            preds = preds[~nans]
        else:
            raise ValueError(f"Invalid option `{target_nan_mask}`")

        loss = loss_fun(preds, targets, reduction="mean")

        return loss

    def get_metrics_logs(
        self, preds: torch.Tensor, targets: torch.Tensor, step_name: str, loss_name: str
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

            loss_name:
                Name of the loss to display in tensorboard

        Returns:
            A dictionary with the keys value being:

            - `loss_name`: The value of the loss
            - `"log"`: A dictionary of type `Dict[str, torch.Tensor]`
                containing the metrics to log on tensorboard.
        """

        targets = targets.to(dtype=self._dtype, device=self._device)
        loss = self.compute_loss(
            preds=preds, targets=targets, target_nan_mask=target_nan_mask, loss_fun=loss_fun
        )

        # Compute the metrics always used in regression tasks
        tensorboard_logs = {f"{self.loss_fun._get_name()}/{step_name}": loss}
        tensorboard_logs[f"mean_pred/{step_name}"] = torch.mean(preds)
        tensorboard_logs[f"std_pred/{step_name}"] = torch.std(preds)

        # Compute the additional metrics
        # TODO: Change this to use metrics on Torch, not numpy
        # TODO: NaN mask `target_nan_mask` not implemented here
        preds2 = preds.clone().cpu().detach()
        targets2 = targets.clone().cpu().detach()
        for key, metric in self.metrics.items():
            metric_name = f"{key}/{step_name}"
            try:
                tensorboard_logs[metric_name] = torch.from_numpy(np.asarray(metric(preds2, targets2)))
            except:
                tensorboard_logs[metric_name] = torch.tensor(float("nan"))
        return {loss_name: loss, "log": tensorboard_logs}

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        *x, y = batch
        preds = self.forward(*x)
        return self.get_metrics_logs(preds=preds, targets=y, step_name="train", loss_name="loss")

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        *x, y = batch
        preds = self.forward(*x)
        return preds, y

    def validation_epoch_end(self, outputs: List):

        # Transform the list of dict of dict, into a dict of list of dict
        preds, targets = zip(*outputs)
        preds = torch.cat(preds, dim=-1)
        targets = torch.cat(targets, dim=-1)
        loss_name = "val_loss"
        loss_logs = self.get_metrics_logs(preds=preds, targets=targets, step_name="val", loss_name=loss_name)
        metrics_names_to_display = [f"{metric_name}/val" for metric_name in self.metrics_on_progress_bar]
        metrics_to_display = {
            metric_name: loss_logs["log"][metric_name] for metric_name in metrics_names_to_display
        }

        self.epoch_summary.set_results(
            name="val",
            predictions=preds,
            targets=targets,
            loss=loss_logs[loss_name],
            metrics=metrics_to_display,
        )
        return loss_logs

    def get_progress_bar_dict(self) -> Dict[str, float]:
        prog_dict = super().get_progress_bar_dict()
        prog_dict["val_loss"] = self.epoch_summary.get_results("val").loss.tolist()
        prog_dict.update(self.epoch_summary.get_results("val").metrics)
        return prog_dict

    def summarize(self, mode: str = ModelSummaryExtended.MODE_DEFAULT) -> ModelSummaryExtended:
        r"""
        Provide a summary of the class, usually to be printed
        """
        model_summary = None

        if mode in ModelSummaryExtended.MODES:
            model_summary = ModelSummaryExtended(self, mode=mode)
            log.info("\n" + str(model_summary))
        elif mode is not None:
            raise MisconfigurationException(
                f"`mode` can be None, {', '.join(ModelSummaryExtended.MODES)}, got {mode}"
            )

        return model_summary

    def __repr__(self) -> str:
        r"""
        Controls how the class is printed
        """
        return self.summarize().__repr__()
