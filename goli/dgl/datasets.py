from multiprocessing import cpu_count
from typing import Union, List, Optional, Any, Iterable, Tuple, Callable

import abc
import dgl
import torch
from torch.utils.data import DataLoader, Dataset
from dgl import DGLGraph
from loguru import logger
from pytorch_lightning import LightningDataModule

from goli.commons.spaces import SPACE_MAP
from goli.commons.utils import is_device_cuda, to_tensor


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
        X: Iterable[Any],
        y: Iterable[Union[float, int]],
        w: Optional[Iterable[float]] = None,
        collate_fn: Optional[Callable] = None,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        r"""
        Parameters:
            X:
                List of input features
            y:
                iterable over the targets associated to the input features
            w:
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
        self.dgl_graphs = dgl_graphs
        self.X = X
        self.y = to_tensor(y)
        self.w = w
        if self.w is not None:
            self.w = to_tensor(self.w)
        self.collate_fn = collate_fn

        # Assert good lengths
        if len(self.X) != len(self.y):
            raise ValueError(f"`X` and `y` must be the same length. Found {len(self.X)}, {len(self.y)}")
        if (self.w is not None) and (len(self.w) != len(self.y)):
            raise ValueError(f"`w` and `y` must be the same length. Found {len(self.w)}, {len(self.y)}")

        # Convert dataset elements to right device and dtype
        self.to(device=self._device, dtype=self._dtype)

    @property
    def has_weights(self) -> bool:
        r"""
        Returns whether the Dataset has weights in `self.w`
        """
        return self.w is not None

    @property
    def dtype(self) -> torch.dtype:
        r"""
        Returns the dtype of the current Dataset
        """
        return self._dtype

    @property
    def is_cuda(self):
        r"""
        Returns whether the Dataset has weights in `self.w`
        """
        return is_device_cuda(self._device)

    @property
    def device(self) -> torch.device:
        r"""
        Returns the device of the current Dataset
        """
        return self._device

    @property
    def collate_fn():
        r"""
        Returns the collate function of the current Dataset
        """
        return self._collate_fn

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current Dataset
        """
        return len(self.X)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Tuple[Any, torch.Tensor, torch.Tensor], Tuple[Any, torch.Tensor]]:
        r"""
        Returns the input featues, the labels, and the weights at the
        specified index.

        Returns:
            X: The input features at index `idx`

            y: Th labels at index `idx`

            w: Th weights at index `idx` if weights are provided. Otherwise
                it is ignored.

        """
        if self.w is not None:
            return self.dgl_graphs[idx], self.y[idx], self.w[idx]
        return self.X[idx], self.y[idx]

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
        self.y = self.y.to(dtype=dtype, device=device)

        # Convert the input feature, if available
        if hasattr(self.X, "to"):
            self.X = self.X.to(dtype=dtype, device=device)
        elif hasattr(self.X[0], "to"):
            self.X = [x.to(dtype=dtype, device=device) for x in self.X]

        # Convert the weights
        if self.w is not None:
            self.w = self.w.to(device)

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
        y: Iterable[Union[float, int]],
        w: Optional[Iterable[float]] = None,
        smiles_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
        device: torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters:
            smiles: iterable of smiles molecules
            y:
                iterable over the targets associated to the input features
            w:
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
        """

        self.smiles = smiles
        self.smiles_transform = smiles_transform
        X = [self.smiles_transform(s) for s in self.smiles]

        super().__init__(
            X=X,
            y=y,
            w=w,
            smiles_transform=smiles_transform,
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
        seed=42,
        train_batch_size=128,
        test_batch_size=256,
        train_shuffle=True,
        eval_shuffle=False,
        collate_fn=None,
        train_sampler=None,
        eval_sampler=None,
        model_device="cpu",
    ):
        super().__init__()

        self.seed = seed
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.collate_fn = collate_fn
        self.train_shuffle = train_shuffle
        self.eval_shuffle = eval_shuffle
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.model_device = model_device
        self.train_dt, self.val_dt, self.test_dt = None, None, None

    def _dataloader(self, dataset, **kwargs):
        data_cuda = dataset[0].is_cuda
        num_workers = self.num_workers if (not data_cuda) else 1
        pin_memory = (not data_cuda) and is_device_cuda(self.model_cuda)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            **kwargs,
        )
        return loader

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self._dataloader(
            dataset=self.train_dt,
            sampler=self.train_sampler,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self._dataloader(
            dataset=self.val_dt,
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


class BaseDGLDataModule(BaseDataModule):
    # Data Processing Code
    def __init__(
        self,
        labels,
        dgl_graphs,
        seed=42,
        train_batch=128,
        test_batch=256,
        train_shuffle=True,
        eval_shuffle=False,
        train_sampler=None,
        eval_sampler=None,
        model_device="cpu",
    ):
        """

        Parameters
        ----------

        labels: torch tensor
            Input target values
        cfg_train: dict
            Training config
        cfg_grid: dict
            CV grid dict. Used for data splitting strategy.

        """
        super().__init__(
            seed=seed,
            train_batch_size=train_batch,
            test_batch_size=test_batch,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
            train_sampler=train_sampler,
            eval_sampler=eval_sampler,
            model_device=model_device,
        )

        self.dgl_graphs = dgl_graphs
        self.labels = labels

    def setup(self, stage: Optional[str] = None):
        """
        Function called by the Pytorch Lightning pipeline to
        generate train, validation, and test datasets.

        Returns
        -------


        """

        self.collate_fn = self._get_collate()
        self.dataset = DGLDataset(self.dgl_graphs, self.labels)
        self.train_dt, self.val_dt, self.test_dt = (
            self.dataset[self.train_ix],
            self.dataset[self.val_ix],
            self.dataset[self.test_ix],
        )
        # Forcing these values to True, else some strange bugs arise. PL fails to set this to true when called manually, and
        # when datamodule is imported from a separate module, but succeeds in setting it when datamodule is defined in same module as
        # train_dl. TODO: Investigate further/raise bug in PL. See PL.trainer.trainer.call_setup_hook.
        self.set_setup_flags()

    def set_setup_flags(self):
        self._has_setup_fit, self._has_setup_test = True, True

    @staticmethod
    def _get_collate():
        collate_fn = DGLCollate(device=None, siamese=False)
        return collate_fn


class DGLFromSmilesDataModule(BaseDGLDataModule):
    # Data Processing Code
    def __init__(
        self,
        labels,
        seed=0,
        n_splits=5,
        splits_method=None,
        splitter_dict=None,
        train_batch=128,
        test_batch=256,
        train_shuffle=True,
        eval_shuffle=False,
        train_sampler=None,
        eval_sampler=None,
    ):
        """

        Parameters
        ----------

        labels: torch tensor
            Input target values
        cfg_train: dict
            Training config
        cfg_grid: dict
            CV grid dict. Used for data splitting strategy.

        """
        super().__init__(
            train_batch_size=train_batch,
            test_batch_size=test_batch,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
            train_sampler=train_sampler,
            eval_sampler=eval_sampler,
        )

        self.seed = seed
        self.n_splits = n_splits
        self.splits_method = splits_method
        if splitter_dict is None:
            splitter_dict = {}
        self.splitter_dict = splitter_dict
        self.dgl_graphs = dgl_graphs
        self.labels = labels

    def setup(self, stage: Optional[str] = None):
        """
        Function called by the Pytorch Lightning pipeline to
        generate train, validation, and test datasets.

        Returns
        -------


        """

        self.collate_fn = self._get_collate()
        self.dataset = DGLDataset(self.dgl_graphs, self.labels)
        self.train_dt, self.val_dt, self.test_dt = (
            self.dataset[self.train_ix],
            self.dataset[self.val_ix],
            self.dataset[self.test_ix],
        )
        # Forcing these values to True, else some strange bugs arise. PL fails to set this to true when called manually, and
        # when datamodule is imported from a separate module, but succeeds in setting it when datamodule is defined in same module as
        # train_dl. TODO: Investigate further/raise bug in PL. See PL.trainer.trainer.call_setup_hook.
        self.set_setup_flags()

    def set_setup_flags(self):
        self._has_setup_fit, self._has_setup_test = True, True

    @staticmethod
    def _get_collate():
        collate_fn = DGLCollate(device=None, siamese=False)
        return collate_fn
