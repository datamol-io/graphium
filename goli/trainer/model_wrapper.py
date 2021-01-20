import os, math
import torch
import dgl
import numpy as np
import pytorch_lightning as pl
from copy import deepcopy

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

from typing import Dict, List, Any, Optional
from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from goli.commons.utils import is_device_cuda
from goli.trainer.reporting import ModelSummaryExtended


class EpochSummary:
    """Container for collecting epoch-wise results"""

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



class ModelWrapper(pl.LightningModule):

    def __init__(self, 
                model: torch.nn.Module,
                dataset: torch.utils.data.Dataset,
                loss_fun,
                lr: float=1e-4,
                batch_size: int=4,
                validation_split: float=0.2,
                random_seed: int=42,
                num_workers: int=0,
                dtype = torch.float32,
                device = 'cpu',
                weight_decay: float= 0.,
                target_nan_mask=None,
                metrics: Dict[str, callable]=None,
                metrics_on_progress_bar=[],
                collate_fn=None,
                additional_hparams=None,
                ):
        r"""

        A class that allows to use regression or classification models easily
        with Pytorch-Lightning.

        Parameters
        ---------------
            model: torch.nn.Module
                Pytorch model trained on the classification/regression task

            dataset: torch.utils.data.Dataset
                The dataset used for the training. 
                If ``validation_split`` is a ``float``, the dataset will be splitted into train/val sets.
                If ``validation_split`` is a ``torch.utils.data.Dataset``, then ``dataset`` variable is the training set,
                and ``validation_split`` is the validation set.

            loss_fun: str, Callable
                Loss function used during training.
                Acceptable strings are 'mse', 'bce', 'mae', 'cosine'.
                Otherwise, a callable object must be provided, with a method ``loss_fun._get_name()``.

            lr: float, Default=1e-4
                The learning rate used during the training.

            batch_size: int, Default=4
                The batch size used during the training.

            validation_split: float, torch.utils.data.Dataset, Default=0.2
                If ``validation_split`` is a ``float``, the ``dataset`` will be splitted into train/val sets.
                The float value must be greater than 0 and smaller than 1.
                If ``validation_split`` is a ``torch.utils.data.Dataset``, then ``dataset`` variable is the training set,
                and ``validation_split`` is the validation set.

            random_seed: int, Default=42
                The random seed used by Pytorch to initialize random tensors.

            num_workers: int, Default=0
                The number of workers used to load the data at each training step.
                ``num_workers`` should be 1 on the GPU
                
                - 0 : Use all cores to load the data in parallel

                - 1: Use a single core to load the data. If using the GPU, this is the only option

                - int: Specify any value for the number of cores to use for data loading.Any

            dtype: torch.dtype, Default=torch.float32
                The desired floating point type of the floating point parameters and buffers in this module.

            device: torch.device, Default='cpu'
                the desired device of the parameters and buffers in this module

            weight_decay: float, Default=0.
                Weight decay used to regularize the optimizer

            target_nan_mask: int, float, str, None: Default=None

                - None: Do not change behaviour if there are nans

                - int, float: Value used to replace nans. For example, if ``target_nan_mask==0``, then
                  all nans will be replaced by zeros

                - 'ignore': Nans will be ignored when computing the loss. NOT YET IMPLEMENTED

            metrics: dict(str, Callable), None, Default=None,
                A dictionnary of metrics to compute on the prediction, other than the loss function.
                These metrics will be logged into TensorBoard.

            metrics_on_progress_bar: list(str), Default=[],
                The metrics names from ``metrics`` to display also on the progress bar of the training

            collate_fn: Callable, Default=None,
                Merges a list of samples to form a mini-batch of Tensor(s). 
                Used when using batched loading from a map-style dataset.
                See ``torch.utils.data.DataLoader.__init__``

            additional_hparams: dict(str, Other), Default=None
                additionnal hyper-parameters to log in the TensorBoard file.
                
        """

        torch.random.manual_seed(random_seed)
        np.random.seed(random_seed)

        super().__init__()
        
        # Basic attributes
        self.model = model.to(dtype=dtype, device=device)
        self.dataset = dataset#.to(dtype=dtype, device=device)
        self.loss_fun = self._parse_loss_fun(loss_fun)
        self.lr = lr
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.num_workers = num_workers   
        self.weight_decay = weight_decay
        self.target_nan_mask = target_nan_mask
        self.metrics = metrics if metrics is not None else {}
        self.metrics_on_progress_bar = metrics_on_progress_bar
        self.collate_fn = collate_fn
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.epoch_summary = EpochSummary()
        self._dtype = dtype
        self._device = device

        self.hparams = deepcopy(self.model.hparams) if hasattr(self.model, 'hparams') else {}
        if additional_hparams is not None:
            self.hparams.update(additional_hparams)

        self.hparams.update({
            'lr': self.lr, 
            'loss_fun': self.loss_fun._get_name(),
            'random_seed': self.random_seed,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'n_params': self.n_params,
            })
        

        self.to(dtype=dtype, device=device)


    @staticmethod
    def _parse_loss_fun(loss_fun):

        # Parse the loss function
        if isinstance(loss_fun, str):
            loss_fun_name = loss_fun.lower()
            if loss_fun_name == 'mse':
                loss_fun = torch.nn.MSELoss()
            elif loss_fun_name == 'bce':
                loss_fun = torch.nn.BCELoss()
            elif (loss_fun_name == 'mae') or (loss_fun_name == 'l1'):
                loss_fun = torch.nn.L1Loss()
            elif loss_fun_name == 'cosine':
                loss_fun = torch.nn.CosineEmbeddingLoss()
            else:
                raise ValueError(f'Unsupported `loss_fun_name`: {loss_fun_name}')
        elif callable(loss_fun):
            pass
        else:
            raise ValueError(f'`loss_fun` must be `str` or `callable`. Provided: {type(loss_fun)}')

        return loss_fun


    def forward(self, *inputs: List[torch.Tensor]):
        """
        """

        out = self.model(*inputs)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out


    def prepare_data(self):
        
        # Creating data indices for training and validation splits:
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))

        if isinstance(self.validation_split, float):
            split = int(np.floor(self.validation_split * dataset_size))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]
            self.val_dataset = self.dataset
            
            # Creating data samplers and loaders:
            self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
        
        elif isinstance(self.validation_split, Dataset):
            train_indices = list(range(dataset_size))
            val_indices = list(range(len(self.validation_split)))
            self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
            self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
            self.val_dataset = self.validation_split

        else:
            raise ValueError('Unsupported validation split')


    def train_dataloader(self):
        num_workers = self.num_workers 
        dataset_cuda = is_device_cuda(self.dataset.device)
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, 
                num_workers=num_workers, sampler = self.train_sampler, drop_last=True, 
                collate_fn=self.collate_fn, pin_memory=dataset_cuda)
        print('\n---------------------\ntrain, dataset_cuda: ', dataset_cuda)
        return train_loader


    def val_dataloader(self):
        num_workers = self.num_workers
        dataset_cuda = is_device_cuda(self.dataset.device)
        valid_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, 
                num_workers=num_workers, sampler = self.valid_sampler, drop_last=True, 
                collate_fn=self.collate_fn, pin_memory=dataset_cuda)
        print('\n---------------------\nval, dataset_cuda: ', dataset_cuda)
        return valid_loader

    
    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=.5)
        return optimiser


    def _compute_loss(self, y_pred, y_true):
        if self.target_nan_mask is None:
            pass
        elif isinstance(self.target_nan_mask, (int, float)):
            y_true[torch.isnan(y_true)] = self.target_nan_mask
        elif self.target_nan_mask == 'ignore':
            raise NotImplementedError('Option `ignore` not yet implemented')
        else:
            raise ValueError(f'Invalid option `{self.target_nan_mask}`')

        loss = self.loss_fun(y_pred, y_true)

        return loss


    def _get_loss_logs(self, y_pred, y_true, step_name:str, loss_name:str):

        y_true = y_true.to(dtype=self._dtype, device=self._device)
        loss = self._compute_loss(y_pred, y_true)

        # Compute the metrics always used in regression tasks
        tensorboard_logs = {f'{self.loss_fun._get_name()}/{step_name}': loss}
        tensorboard_logs[f'mean_pred/{step_name}'] = torch.mean(y_pred)
        tensorboard_logs[f'std_pred/{step_name}'] = torch.std(y_pred)

        # Compute the additional metrics
        y_pred2 = y_pred.clone().cpu().detach()
        y_true2 = y_true.clone().cpu().detach()
        for key, metric in self.metrics.items():
            metric_name = f'{key}/{step_name}'
            try:
                tensorboard_logs[metric_name] = torch.from_numpy(np.asarray(metric(y_pred2, y_true2)))
            except:
                tensorboard_logs[metric_name] = torch.tensor(float('nan'))
        return {loss_name: loss, 'log': tensorboard_logs}


    def _get_val_logs(self, y_pred, y_true, step_name:str):
        return self._get_loss_logs(y_pred=y_pred, y_true=y_true, step_name=step_name, loss_name='val_loss')


    def training_step(self, batch, batch_idx):
        *x, y = batch
        preds = self.forward(*x)
        return self._get_loss_logs(y_pred=preds, y_true=y, step_name='train', loss_name='loss')

 
    def validation_step(self, batch, batch_idx):
        *x, y = batch
        preds = self.forward(*x)
        return preds, y


    def validation_epoch_end(self, outputs):
        # Transform the list of dict of dict, into a dict of list of dict
        y_pred, y_true = zip(*outputs)
        y_pred = torch.cat(y_pred, dim=-1)
        y_true = torch.cat(y_true, dim=-1)
        loss_name='val_loss'
        loss_logs = self._get_loss_logs(y_pred=y_pred, y_true=y_true, step_name='val', loss_name=loss_name)
        metrics_names_to_display = [f'{metric_name}/val' for metric_name in self.metrics_on_progress_bar]
        metrics_to_display = {metric_name: loss_logs['log'][metric_name] for metric_name in metrics_names_to_display}

        self.epoch_summary.set_results(name='val', predictions=y_pred, targets=y_true, loss=loss_logs[loss_name], metrics=metrics_to_display)
        return loss_logs


    def get_progress_bar_dict(self):
        prog_dict = super().get_progress_bar_dict()
        prog_dict['val_loss'] = self.epoch_summary.get_results('val').loss.tolist()
        prog_dict.update(self.epoch_summary.get_results('val').metrics)
        return prog_dict


    def summarize(self, mode: Optional[str] = ModelSummaryExtended.MODE_DEFAULT) -> Optional[ModelSummaryExtended]:
        model_summary = None

        if mode in ModelSummaryExtended.MODES:
            model_summary = ModelSummaryExtended(self, mode=mode)
            log.info("\n" + str(model_summary))
        elif mode is not None:
            raise MisconfigurationException(
                f"`mode` can be None, {', '.join(ModelSummaryExtended.MODES)}, got {mode}"
            )

        return model_summary
        