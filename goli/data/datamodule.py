from typing import List, Dict, Union, Any, Callable, Optional

import os
import functools
import copy
import yaml

from loguru import logger
import fsspec
import omegaconf

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import torch.utils.data
import dgl
import pytorch_lightning as pl

import datamol as dm

from goli.utils import fs
from goli.features import mol_to_dglgraph
from goli.data.collate import goli_collate_fn


class DGLFromSmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: List[str], features: List[dgl.DGLGraph], labels: torch.Tensor):
        self.smiles = smiles
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        datum = {}
        datum["smiles"] = self.smiles[idx]
        datum["features"] = self.features[idx]
        datum["labels"] = self.labels[idx]
        return datum


class DGLFromSmilesDataModule(pl.LightningDataModule):
    """
    NOTE(hadim): let's make only one class for the moment and refactor with a parent class
    once we have more concrete datamodules to implement. The class should be general enough
    to be easily refactored.

    NOTE(hadim): splitting is not very full-featured yet; only random splitting on-the-fly
    is allowed using a seed. Next is to add the availability to provide split indices data as input.

    NOTE(hadim): implement using weights. It should probably be a column in the dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        df_path: Optional[Union[str, os.PathLike]] = None,
        cache_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        smiles_col: str = None,
        label_cols: List[str] = None,
        split_val: float = 0.2,
        split_test: float = 0.2,
        split_seed: int = None,
        splits_path: Optional[Union[str, os.PathLike]] = None,
        batch_size_train_val: int = 16,
        batch_size_test: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        collate_fn: Optional[Callable] = None,
    ):
        """

        Parameters:
            df: a dataframe.
            df_path: a path to a dataframe to load (CSV file). `df` takes precedence over
                `df_path`.
            cache_data_path: path where to save or reload the cached data. The path can be
                remote (S3, GS, etc).
            featurization: args to apply to the SMILES to DGL featurizer.
            smiles_col: Name of the SMILES column. If set to `None`, it will look for
                a column with the word "smile" (case insensitive) in it.
                If no such column is found, an error will be raised.
            label_cols: Name of the columns to use as labels. If set to None, all the
                columns are used except the SMILES one.
            split_val: Ratio for the validation split.
            split_test: Ratio for the test split.
            split_seed: Seed to use for the random split. More complex splitting strategy
                should be implemented.
            splits_path: A path a CSV file containing indices for the splits. The file must contains
                3 columns "train", "val" and "test". It takes precedence over `split_val` and `split_test`.
            batch_size_train_val: batch size for training and val dataset.
            batch_size_test: batch size for test dataset.
            num_workers: Number of workers for the dataloader. Use -1 to use all available
                cores.
            pin_memory: Whether to pin on paginated CPU memory for the dataloader.
            featurization_n_jobs: Number of cores to use for the featurization.
            featurization_progress: whether to show a progress bar during featurization.
            collate_fn: A custom torch collate function. Default is to `goli.data.goli_collate_fn`
        """
        super().__init__()

        self.df = df
        self.df_path = df_path

        self.cache_data_path = str(cache_data_path) if cache_data_path is not None else None
        self.featurization = featurization

        self.smiles_col = smiles_col
        self.label_cols = label_cols
        self.split_val = split_val
        self.split_test = split_test
        self.split_seed = split_seed
        self.splits_path = splits_path

        self.batch_size_train_val = batch_size_train_val
        self.batch_size_test = batch_size_test

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.featurization_n_jobs = featurization_n_jobs
        self.featurization_progress = featurization_progress

        if collate_fn is None:
            self.collate_fn = goli_collate_fn
        else:
            self.collate_fn = collate_fn

        self.ds = None
        self.train_ds = None
        self.val_ds = None
        self.ds_ds = None

    def prepare_data(self):
        """Called only from a single process in distributed settings. Steps:

        - If cache is set and exists, reload from cache.
        - Load the dataframe if its a path.
        - Extract smiles and labels from the dataframe.
        - Compute the features.
        - Compute or set split indices.
        - Make the list of dict dataset.
        """

        # Reload from cache if it exists and is valid
        if self.cache_data_path is not None and fs.exists(self.cache_data_path):
            logger.info(f"Reload data from {self.cache_data_path}.")

            with fsspec.open(self.cache_data_path, "rb") as f:
                cache = torch.load(f)  # type: ignore

            if set(cache.keys()) == {"dataset", "test_indices", "train_indices", "val_indices"}:
                self.dataset = cache["dataset"]
                self.train_indices = cache["train_indices"]
                self.val_indices = cache["val_indices"]
                self.test_indices = cache["test_indices"]
                return
            else:
                logger.info(f"Cache looks invalid with keys: {cache.keys()}")
                logger.info("Fallback to regular data preparation steps.")

        # Load the dataframe
        if self.df is None:
            df = pd.read_csv(self.df_path)
        else:
            df = self.df

        logger.info(f"Prepare dataset with {len(df)} data points.")

        # Extract smiles and labels
        smiles, labels = self._extract_smiles_labels(df, self.smiles_col, self.label_cols)

        # Precompute the features
        # NOTE(hadim): in case of very large dataset we could:
        # - or cache the data and read from it during `next(iter(dataloader))`
        # - or compute the features on-the-fly during `next(iter(dataloader))`
        # For now we compute in advance and hold everything in memory.
        featurization_args = self.featurization or {}
        transform_smiles = functools.partial(mol_to_dglgraph, **featurization_args)
        features = dm.utils.parallelized(
            transform_smiles,
            smiles,
            progress=self.featurization_progress,
            n_jobs=self.featurization_n_jobs,
        )

        # Get splits indices
        self.train_indices, self.val_indices, self.test_indices = self._get_split_indices(
            len(df),
            split_val=self.split_val,
            split_test=self.split_test,
            split_seed=self.split_seed,
            splits_path=self.splits_path,
        )

        # Make the torch datasets (mostly a wrapper there is no memory overhead here)
        self.dataset = DGLFromSmilesDataset(smiles=smiles, features=features, labels=labels)

        # Cache on disk
        if self.cache_data_path is not None:
            logger.info(f"Write prepared data to {self.cache_data_path}")
            cache = {}
            cache["dataset"] = self.dataset
            cache["train_indices"] = self.train_indices
            cache["val_indices"] = self.val_indices
            cache["test_indices"] = self.test_indices
            with fsspec.open(self.cache_data_path, "wb") as f:
                torch.save(cache, f)  # type: ignore

    def setup(self, stage: str = None):
        """Prepare the torch dataset. Called on every GPUs. Setting state here is ok."""

        if stage == "fit" or stage is None:
            self.train_ds = torch.utils.data.Subset(self.dataset, self.train_indices)
            self.val_ds = torch.utils.data.Subset(self.dataset, self.val_indices)

        if stage == "test" or stage is None:
            self.test_ds = torch.utils.data.Subset(self.dataset, self.test_indices)

    def train_dataloader(self, **kwargs):
        return self._dataloader(
            dataset=self.train_ds,  # type: ignore
            batch_size=self.batch_size_train_val,
            shuffle=True,
        )

    def val_dataloader(self, **kwargs):
        return self._dataloader(
            dataset=self.val_ds,  # type: ignore
            batch_size=self.batch_size_train_val,
            shuffle=False,
        )

    def test_dataloader(self, **kwargs):

        return self._dataloader(
            dataset=self.test_ds,  # type: ignore
            batch_size=self.batch_size_test,
            shuffle=False,
        )

    @property
    def is_prepared(self):
        if not hasattr(self, "dataset"):
            return False
        return getattr(self, "dataset") is not None

    @property
    def is_setup(self):
        if not hasattr(self, "train_ds"):
            return False
        return getattr(self, "train_ds") is not None

    @property
    def num_node_feats(self):
        """Return the number of node features in the first graph"""

        graph = self.get_first_graph()
        if "feat" in graph.ndata.keys():
            return graph.ndata["feat"].shape[1]  # type: ignore
        else:
            return 0

    @property
    def num_edge_feats(self):
        """Return the number of edge features in the first graph"""

        graph = self.get_first_graph()
        if "feat" in graph.edata.keys():
            return graph.edata["feat"].shape[1]  # type: ignore
        else:
            return 0

    def get_first_graph(self):
        """Low memory footprint method to get the first datapoint DGL graph."""
        if self.df is None:
            df = pd.read_csv(self.df_path, nrows=1)
        else:
            df = self.df.iloc[0:1, :]

        smiles, _ = self._extract_smiles_labels(df, self.smiles_col, self.label_cols)

        featurization_args = self.featurization or {}
        transform_smiles = functools.partial(mol_to_dglgraph, **featurization_args)
        graph = transform_smiles(smiles[0])
        return graph

    # Private methods

    def _dataloader(self, dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool):
        """Get a dataloader for a given dataset"""

        if self.num_workers == -1:
            num_workers = os.cpu_count()
            num_workers = num_workers if num_workers is not None else 0
        else:
            num_workers = self.num_workers

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return loader

    def _extract_smiles_labels(self, df: pd.DataFrame, smiles_col: str = None, label_cols: List[str] = None):
        """For a given dataframe extract the SMILES and labels columns. Smiles is returned as a list
        of string while labels are returned as a 2D numpy array.
        """

        if smiles_col is None:
            smiles_col = [col for col in df.columns if "smile" in str(col).lower()][0]

        if label_cols is None:
            label_cols = df.columns.drop(smiles_col)

        smiles = df[smiles_col].to_list()
        labels = df[label_cols].values

        return smiles, labels

    def _get_split_indices(
        self,
        dataset_size: int,
        split_val: float,
        split_test: float,
        split_seed: int = None,
        splits_path: Union[str, os.PathLike] = None,
    ):
        """Compute indices of random splits."""

        if splits_path is None:
            # Random splitting
            indices = np.arange(dataset_size)
            train_indices, val_test_indices = train_test_split(
                indices,
                test_size=split_val + split_test,
                random_state=split_seed,
            )

            sub_split_test = split_test / (split_test + split_val)
            val_indices, test_indices = train_test_split(
                val_test_indices,
                test_size=sub_split_test,
                random_state=split_seed,
            )

            train_indices = list(train_indices)
            val_indices = list(val_indices)
            test_indices = list(test_indices)

        else:
            # Split from an indices file
            with fsspec.open(str(splits_path)) as f:
                splits = pd.read_csv(splits_path)

            train_indices = splits["train"].dropna().astype("int").tolist()
            val_indices = splits["val"].dropna().astype("int").tolist()
            test_indices = splits["test"].dropna().astype("int").tolist()

        return train_indices, val_indices, test_indices

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current DataModule
        """
        if self.df is None:
            df = pd.read_csv(self.df_path)
        else:
            df = self.df

        return len(df)

    def __repr__(self):
        """Controls how the class is printed"""
        obj_repr = {}
        obj_repr["name"] = self.__class__.__name__
        obj_repr["len"] = len(self)
        obj_repr["batch_size_train_val"] = self.batch_size_train_val
        obj_repr["batch_size_test"] = self.batch_size_test
        obj_repr["num_node_feats"] = self.num_node_feats
        obj_repr["num_edge_feats"] = self.num_edge_feats
        obj_repr["collate_fn"] = self.collate_fn.__name__
        obj_repr["featurization"] = self.featurization

        obj_str = omegaconf.OmegaConf.to_yaml(obj_repr)
        return obj_str
