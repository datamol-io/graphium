from multiprocessing import cpu_count
from typing import Union, List, Optional

import dgl
import torch
from torch.utils.data import DataLoader
from dgl import DGLGraph
from loguru import logger
from pytorch_lightning import LightningDataModule

from goli.commons.spaces import SPACE_MAP
from goli.commons.utils import is_device_cuda


# from invivoplatform.automation.utils.processing_utils import (
#     _generate_train_test_split_idx,
# )
# from invivoplatform.automation.utils.ml_utils import extract_train_test_df


CPUS = cpu_count()


class SmilesDataset(torch.utils.data.Dataset):

    """
    A simple dataset for working with Smiles data that allows for dynamic featurization over the Smiles.
    """

    def __init__(self, smiles, labels, smiles_transform=None):
        """

        :param smiles: iterable of smiles molecules
        :param labels: iterable over targets (multiple targets assumed)
        :param smiles_transform: Feature Transformation to apply over the smiles, for example FingerprintsTransformer.
        """

        if not isinstance(labels, torch.Tensor):
            try:
                # Assuming list of targets
                assert all(len(labels[0]) == len(l) for l in labels)
                labels = torch.tensor(labels).reshape(len(labels[0]), -1)
            except TypeError as e:
                # Assuming single array of labels (one target)
                labels = torch.tensor(labels).reshape(len(labels), -1)
        self.smiles = np.array(smiles)
        self.labels = labels.float()

        assert len(self.smiles) == self.labels.shape[0]

        self.transform = smiles_transform

    def __getitem__(self, idx):
        if self.transform is None:
            return self.smiles[idx], self.labels[idx].float()
        elif isinstance(idx, int):
            return self.transform(self.smiles[idx]), self.labels[idx].float()
        elif torch.is_tensor(idx) and idx.dtype == torch.long:
            return self.transform(self.smiles[idx]), self.labels[idx].float()
        else:
            raise IndexError(
                "Only integers and long are valid " "indices (got {}).".format(type(idx).__name__)
            )

    def __len__(self):
        return len(self.smiles)


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
        train_batch_size=128,
        test_batch_size=256,
        train_shuffle=True,
        eval_shuffle=False,
        collate_fn=None,
        train_sampler=None,
        eval_sampler=None,
        training_device="cpu",
    ):
        super().__init__()

        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.collate_fn = collate_fn
        self.train_shuffle = train_shuffle
        self.eval_shuffle = eval_shuffle
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.training_device = training_device
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
            shuffle=self.eval_shuffle,
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


class DGLFromSmilesDataModule(DGLDataModule):
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
