from typing import Union, List, Optional, Any, Iterable, Callable

from functools import partial

import dgl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from goli.utils.utils import is_device_cuda
from goli.features.featurizer import mol_to_dglgraph

from .dataset import SmilesDataset


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
        train_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
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
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
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
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self._dataloader(
            dataset=self.dataset,
            sampler=self.val_sampler,
            batch_size=self.train_batch_size,
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:

        return self._dataloader(
            dataset=self.dataset,
            sampler=self.test_sampler,
            batch_size=self.test_batch_size,
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
        train_sampler: Optional[Callable] = None,
        val_sampler: Optional[Callable] = None,
        test_sampler: Optional[Callable] = None,
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
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            test_sampler=test_sampler,
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
