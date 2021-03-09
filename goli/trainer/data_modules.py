import numpy as np
import pytorch_lightning as pl

# Torch
import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import Dataset

from goli.commons.utils import is_device_cuda


class DGLDataModule(pl.LightningDataModule):

    def __init__(self, 
            data_dir: str, 
            train_split_or_subdir: Union[str, float],
            val_split_or_subdir: Union[str, float],
            test_split_or_subdir: Union[str, float],
            num_workers: int = 0,
            train_batch_size: int = 128,
            test_batch_size: int = 128,
            seed: int = 42,
            ):
        super().__init__()

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_split_or_subdir = train_split_or_subdir
        self.val_split_or_subdir = val_split_or_subdir
        self.test_split_or_subdir = test_split_or_subdir
        self.seed = seed


    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)


    def setup(self, stage=None):
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

        if stage is None:
            pass
        if stage == 'fit':
            pass
        if stage == 'test':
            pass

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

    def _dataloader(self, dataset, batch_size, sampler):
        num_workers = self.num_workers
        dataset_cuda = is_device_cuda(dataset.device)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True,
            collate_fn=self.collate_fn,
            pin_memory=dataset_cuda,
        )
        return loader

    def train_dataloader(self):
        return self._dataloader(dataset=self.dataset, batch_size=self.train_batch_size, sampler=self.train_sampler)

    def val_dataloader(self):
        return self._dataloader(dataset=self.val_dataset, batch_size=self.train_batch_size, sampler=self.val_sampler)

    def test_dataloader(self):
        return self._dataloader(dataset=self.test_dataset, batch_size=self.test_batch_size, sampler=self.test_sampler)
