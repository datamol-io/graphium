from multiprocessing import cpu_count
from typing import Union, List, Optional, Any, Iterable, Tuple, Callable

import abc
import dgl
import torch
from torch.utils.data import DataLoader, Dataset
from dgl import DGLGraph
from loguru import logger
from pytorch_lightning import LightningDataModule
from functools import partial

import datamol as dm

from goli.commons.spaces import SPACE_MAP
from goli.commons.utils import is_device_cuda, to_tensor
from goli.mol_utils.featurizer import mol_to_dglgraph


# from invivoplatform.automation.utils.processing_utils import (
#     _generate_train_test_split_idx,
# )
# from invivoplatform.automation.utils.ml_utils import extract_train_test_df


CPUS = cpu_count()


class BaseDataset(Dataset):
    r"""
    A base class to use to define the different datasets
    """

    def __init__(
        self,
        feats: Iterable[Any],
        labels: Iterable[Union[float, int]],
        weights: Optional[Iterable[float]] = None,
        collate_fn: Optional[Callable] = None,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        r"""
        Parameters:
            feats:
                List of input features
            labels:
                iterable over the targets associated to the input features
            weights:
                sample or prediction weights to be used in a cost function
            collate_fn:
                The collate function used to batch multiple input features.
                If None, the default collate function is used.
            device:
                The device on which to run the computation
            dtype:
                The torch dtype for the data

        """

        super().__init__()

        # simple attributes
        self.feats = feats
        self.labels = to_tensor(labels)
        self.weights = weights
        if self.weights is not None:
            self.weights = to_tensor(self.weights)
        self._collate_fn = collate_fn

        # Assert good lengths
        if len(self.feats) != len(self.labels):
            raise ValueError(
                f"`feats` and `labels` must be the same length. Found {len(self.feats)}, {len(self.labels)}"
            )
        if (self.weights is not None) and (len(self.weights) != len(self.labels)):
            raise ValueError(
                f"`weights` and `labels` must be the same length. Found {len(self.weights)}, {len(self.labels)}"
            )

        # Convert dataset elements to right device and dtype
        self.to(device=device, dtype=dtype)

    @property
    def has_weights(self) -> bool:
        r"""
        Returns whether the Dataset has weights in `self.weights`
        """
        return self.weights is not None

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns the dtype of the current Dataset
        """
        return self._dtype

    @property
    def is_cuda(self):
        r"""
        Returns whether the Dataset has weights in `self.weights`
        """
        return is_device_cuda(self._device)

    @property
    def device(self) -> torch.device:
        r"""
        Returns the device of the current Dataset
        """
        return self._device

    @property
    def collate_fn(self) -> Callable:
        r"""
        Returns the collate function of the current Dataset
        """
        return self._collate_fn

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current Dataset
        """
        return len(self.feats)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Tuple[Any, torch.Tensor, torch.Tensor], Tuple[Any, torch.Tensor]]:
        r"""
        Returns the input featues, the labels, and the weights at the
        specified index.

        Returns:
            feats: The input features at index `idx`

            labels: Th labels at index `idx`

            weights: Th weights at index `idx` if weights are provided. Otherwise
                it is ignored.

        """
        if self.has_weights:
            return self.feats[idx], self.labels[idx], self.weights[idx]
        return self.feats[idx], self.labels[idx]

    def to(self, device: torch.device = None, dtype: torch.dtype = None):
        r"""
        Convert the stored data to the specified device and dtype

        Parameters:
            device:
                The device on which to run the computation.
                If None, the current device is not changed
            dtype:
                The torch dtype for the data.
                If None, the current type is not changed

        Returns:
            New `BaseDataset`

        """

        # Set the attributes
        if device is not None:
            self._device = torch.device(device)
        if dtype is not None:
            self._dtype = dtype

        # Convert the labels
        self.labels = self.labels.to(dtype=dtype, device=device)

        # Convert the input feature, if available
        if hasattr(self.feats, "to"):
            self.feats = self.feats.to(dtype=dtype, device=device)
        elif hasattr(self.feats[0], "to"):
            self.feats = [x.to(dtype=dtype, device=device) for x in self.feats]

        # Convert the weights
        if self.weights is not None:
            self.weights = self.weights.to(device)

        return self


class SmilesDataset(BaseDataset):

    """
    A simple dataset for working with Smiles data and apply any
    kind of smiles transformer to it, such as a transformer that
    returns a `DGL.graph` object.
    """

    def __init__(
        self,
        smiles: Iterable[str],
        labels: Iterable[Union[float, int]],
        weights: Optional[Iterable[float]] = None,
        smiles_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        n_jobs: int = -1,
    ):
        """
        Parameters:
            smiles: iterable of smiles molecules
            labels:
                iterable over the targets associated to the input features
            weights:
                sample or prediction weights to be used in a cost function
            smiles_transform:
                Feature Transformation to apply over the smiles,
                for example fingerprints transformers or dgl graphs transformers.
            collate_fn:
                The collate function used to batch multiple input features.
                If None, the default collate function is used.
            device:
                The device on which to run the computation
            dtype:
                The torch dtype for the data
            n_jobs:
                The number of jobs to use for the smiles transform
        """

        self.n_jobs = n_jobs
        self.smiles = smiles
        self.smiles_transform = smiles_transform

        runner = dm.JobRunner(n_jobs=n_jobs, progress=True, prefer="threads")
        if self.smiles_transform is None:
            feats = self.smiles
        else:
            feats = runner(callable_fn=self.smiles_transform, data=self.smiles)

        super().__init__(
            feats=feats,
            labels=labels,
            weights=weights,
            collate_fn=collate_fn,
            device=device,
            dtype=dtype,
        )


class DGLCollate:
    def __init__(self, device, siamese=False):
        self.device = device
        self.siamese = siamese

    def __call__(self, samples):
        # The input `samples` is a list of pairs
        #  (graph, label).

        graphs, labels = map(list, zip(*samples))
        if isinstance(labels[0], torch.Tensor):
            labels = torch.stack(labels, dim=0)

        if (isinstance(graphs[0], (tuple, list))) and (len(graphs[0]) == 1):
            graphs = [g[0] for g in graphs]

        if self.siamese:
            graphs1, graphs2 = zip(*graphs)
            batched_graph = (dgl.batch(graphs1).to(self.device), dgl.batch(graphs2).to(self.device))
        else:
            batched_graph = dgl.batch(graphs).to(self.device)
        return batched_graph, labels


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        labels: Iterable[Union[float, int]],
        weights: Optional[Iterable[float]] = None,
        seed: int = 42,
        train_batch_size: int = 128,
        test_batch_size: int = 256,
        train_shuffle: bool = True,
        eval_shuffle: bool = False,
        train_sampler: Optional[Callable] = None,
        eval_sampler: Optional[Callable] = None,
        model_device: Union[str, torch.device] = "cpu",
        data_device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        n_jobs: int = -1,
    ):
        super().__init__()

        self.labels = labels
        self.weights = weights
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.eval_shuffle = eval_shuffle
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.model_device = torch.device(model_device)
        self.data_device = torch.device(data_device)
        self.dtype = dtype
        self.n_jobs = n_jobs if n_jobs >= 0 else CPUS
        self.dataset = None

    def _dataloader(self, *args, **kwargs):
        r"""
        Regrouping the logic behind the number of workers and pin_memory
        for all data loaders.
        """
        data_cuda = is_device_cuda(self.data_device)
        num_workers = self.n_jobs if (not data_cuda) else 0
        pin_memory = (not data_cuda) and is_device_cuda(self.model_device)

        loader = torch.utils.data.DataLoader(
            num_workers=n_jobs,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            *args,
            **kwargs,
        )
        return loader

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._dataloader(
            dataset=self.dataset,
            sampler=self.train_sampler,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self._dataloader(
            dataset=self.dataset,
            sampler=self.eval_sampler,
            batch_size=self.train_batch_size,
            shuffle=self.eval_shuffle,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:

        return self._dataloader(
            dataset=self.test_dt,
            sampler=self.eval_sampler,
            batch_size=self.test_batch_size,
            shuffle=self.eval_shuffle,
        )

    @property
    def collate_fn(self):
        r"""
        Returns the collate function of the current Dataset
        """
        return self.dataset.collate_fn

    @property
    def has_weights(self) -> bool:
        r"""
        Returns whether the Dataset has weights in `self.weights`
        """
        return self.dataset.has_weights


class DGLFromSmilesDataModule(BaseDataModule):
    r"""
    DataModule for DGL graphs created from a dataste of SMILES
    """

    def __init__(
        self,
        smiles: Iterable[str],
        labels: Iterable[Union[float, int]],
        weights: Optional[Iterable[float]] = None,
        seed: int = 42,
        train_batch_size: int = 128,
        test_batch_size: int = 256,
        train_shuffle: bool = True,
        eval_shuffle: bool = False,
        train_sampler: Optional[Callable] = None,
        eval_sampler: Optional[Callable] = None,
        model_device: Union[str, torch.device] = "cpu",
        data_device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        n_jobs: int = -1,
        smiles_transform: Callable = partial(mol_to_dglgraph, atom_property_list_onehot=["atomic-number"]),
    ):
        r"""
        TODO: docs
        """
        super().__init__(
            labels=labels,
            weights=weights,
            seed=seed,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
            train_sampler=train_sampler,
            eval_sampler=eval_sampler,
            model_device=model_device,
            data_device=data_device,
            dtype=dtype,
            n_jobs=n_jobs,
        )

        self.smiles = smiles
        self.smiles_transform = smiles_transform
        self.dataset = None

    def setup(self, stage: Optional[str] = None):
        r"""
        Function called by the Pytorch Lightning pipeline to
        generate train, validation, and test datasets.
        """

        self.dataset = SmilesDataset(
            smiles=self.smiles,
            labels=self.labels,
            weights=self.weights,
            smiles_transform=self.smiles_transform,
            collate_fn=DGLCollate(device=None, siamese=False),
            device=self.data_device,
            dtype=self.dtype,
            n_jobs=self.n_jobs,
        )

        # Forcing these values to True, else some strange bugs arise. PL fails to set this to true when called manually, and
        # when datamodule is imported from a separate module, but succeeds in setting it when datamodule is defined in same module as
        # train_dl. TODO: Investigate further/raise bug in PL. See PL.trainer.trainer.call_setup_hook.
        self.set_setup_flags()

    def set_setup_flags(self):
        self._has_setup_fit, self._has_setup_test = True, True
