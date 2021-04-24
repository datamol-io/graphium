from typing import List, Dict, Union, Any, Callable, Optional

import os
import functools
import importlib.resources
import zipfile

from loguru import logger
import fsspec
import omegaconf

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import dgl
import pytorch_lightning as pl

import datamol as dm

from goli.utils import fs
from goli.features import mol_to_dglgraph
from goli.data.collate import goli_collate_fn


import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import Subset


class DGLDataset(Dataset):
    def __init__(
        self,
        features: List[dgl.DGLGraph],
        labels: Union[torch.Tensor, np.ndarray],
        smiles: Optional[List[str]] = None,
        indices: Optional[List[str]] = None,
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        self.smiles = smiles
        self.features = features
        self.labels = labels
        self.indices = indices
        self.weights = weights

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        datum = {}

        if self.smiles is not None:
            datum["smiles"] = self.smiles[idx]

        if self.indices is not None:
            datum["indices"] = self.indices[idx]

        if self.weights is not None:
            datum["weights"] = self.weights[idx]

        datum["features"] = self.features[idx]
        datum["labels"] = self.labels[idx]
        return datum


class DGLBaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size_train_val: int = 16,
        batch_size_test: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        collate_fn: Optional[Callable] = None,
    ):
        super().__init__()

        self.batch_size_train_val = batch_size_train_val
        self.batch_size_test = batch_size_test

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if collate_fn is None:
            self.collate_fn = goli_collate_fn
        else:
            self.collate_fn = collate_fn

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def prepare_data(self):
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

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
        raise NotImplementedError()

    @property
    def is_setup(self):
        raise NotImplementedError()

    @property
    def num_node_feats(self):
        raise NotImplementedError()

    @property
    def num_edge_feats(self):
        raise NotImplementedError()

    def get_first_graph(self):
        raise NotImplementedError()

    # Private methods

    def _dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool):
        """Get a dataloader for a given dataset"""

        if self.num_workers == -1:
            num_workers = os.cpu_count()
            num_workers = num_workers if num_workers is not None else 0
        else:
            num_workers = self.num_workers

        loader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers,
        )
        return loader


class DGLFromSmilesDataModule(DGLBaseDataModule):
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
        weights_col: str = None,
        weights_type: str = None,
        idx_col: str = None,
        split_val: float = 0.2,
        split_test: float = 0.2,
        split_seed: int = None,
        splits_path: Optional[Union[str, os.PathLike]] = None,
        batch_size_train_val: int = 16,
        batch_size_test: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
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
            weights_col: Name of the column to use as sample weights. If `None`, no
                weights are used. This parameter cannot be used together with `weights_type`.
            weights_type: The type of weights to use. This parameter cannot be used together with `weights_col`.
                **It only supports multi-label binary classification.**

                Supported types:

                - `None`: No weights are used.
                - `"sample_balanced"`: A weight is assigned to each sample inversely
                    proportional to the number of positive value. If there are multiple
                    labels, the product of the weights is used.
                - `"sample_label_balanced"`: Similar to the `"sample_balanced"` weights,
                    but the weights are applied to each element individually, without
                    computing the product of the weights for a given sample.

            idx_col: Name of the columns to use as indices. Unused if set to None.
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
        super().__init__(
            batch_size_train_val=batch_size_train_val,
            batch_size_test=batch_size_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

        self.df = df
        self.df_path = df_path

        self.cache_data_path = str(cache_data_path) if cache_data_path is not None else None
        self.featurization = featurization

        self.smiles_col = smiles_col
        self.label_cols = label_cols
        self.idx_col = idx_col

        self.weights_col = weights_col
        self.weights_type = weights_type
        if self.weights_col is not None:
            assert self.weights_type is None

        self.split_val = split_val
        self.split_test = split_test
        self.split_seed = split_seed
        self.splits_path = splits_path

        self.featurization_n_jobs = featurization_n_jobs
        self.featurization_progress = featurization_progress

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

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
        smiles, labels, indices, weights = self._extract_smiles_labels(
            df, self.smiles_col, self.label_cols, self.idx_col
        )

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
        self.dataset = DGLDataset(
            smiles=smiles, features=features, labels=labels, indices=indices, weights=weights
        )

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
            self.train_ds = Subset(self.dataset, self.train_indices)  # type: ignore
            self.val_ds = Subset(self.dataset, self.val_indices)  # type: ignore

        if stage == "test" or stage is None:
            self.test_ds = Subset(self.dataset, self.test_indices)  # type: ignore

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
        num_feats = 0
        if "feat" in graph.ndata.keys():
            num_feats += graph.ndata["feat"].shape[1]
        return num_feats

    @property
    def num_node_feats_with_positional_encoding(self):
        """Return the number of node features in the first graph
        including positional encoding features."""

        graph = self.get_first_graph()
        num_feats = 0
        if "feat" in graph.ndata.keys():
            num_feats += graph.ndata["feat"].shape[1]
        if "pos_enc_feats_sign_flip" in graph.ndata.keys():
            num_feats += graph.ndata["pos_enc_feats_sign_flip"].shape[1]
        if "pos_enc_feats_no_flip" in graph.ndata.keys():
            num_feats += graph.ndata["pos_enc_feats_no_flip"].shape[1]
        return num_feats

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

        smiles, _, _, _ = self._extract_smiles_labels(df, self.smiles_col, self.label_cols)

        featurization_args = self.featurization or {}
        transform_smiles = functools.partial(mol_to_dglgraph, **featurization_args)
        graph = transform_smiles(smiles[0])
        return graph

    # Private methods

    def _extract_smiles_labels(
        self,
        df: pd.DataFrame,
        smiles_col: str = None,
        label_cols: List[str] = None,
        idx_col: str = None,
        weights_col: str = None,
        weights_type: str = None,
    ):
        """For a given dataframe extract the SMILES and labels columns. Smiles is returned as a list
        of string while labels are returned as a 2D numpy array.
        """

        if smiles_col is None:
            smiles_col_all = [col for col in df.columns if "smile" in str(col).lower()]
            if len(smiles_col_all) == 0:
                raise ValueError(f"No SMILES column found in dataframe. Columns are {df.columns}")
            elif len(smiles_col_all) > 1:
                raise ValueError(
                    f"Multiple SMILES column found in dataframe. SMILES Columns are {smiles_col_all}"
                )

            smiles_col = smiles_col_all[0]

        if label_cols is None:
            label_cols = df.columns.drop(smiles_col)

        smiles = df[smiles_col].to_list()
        labels = df[label_cols].values

        indices = None
        if idx_col is not None:
            indices = df[idx_col].to_list()

        # Extract the weights
        weights = None
        if weights_col is not None:
            weights = df[weights_col].values
        elif weights_type is not None:
            if not np.all((labels == 0) | (labels == 1)):
                raise ValueError("Labels must be binary for `weights_type`")

            if weights_type == "sample_label_balanced":
                ratio_pos_neg = np.sum(labels, axis=0, keepdims=1) / labels.shape[0]
                weights = np.zeros(labels.shape)
                weights[labels == 0] = ratio_pos_neg
                weights[labels == 1] = ratio_pos_neg ** -1

            elif weights_type == "sample_balanced":
                ratio_pos_neg = np.sum(labels, axis=0, keepdims=1) / labels.shape[0]
                weights = np.zeros(labels.shape)
                weights[labels == 0] = ratio_pos_neg
                weights[labels == 1] = ratio_pos_neg ** -1
                weights = np.prod(weights, axis=1)

            else:
                raise ValueError(f"Undefined `weights_type` {weights_type}")

            weights /= np.max(weights)  # Put the max weight to 1

        return smiles, labels, indices, weights

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

    def to_dict(self):
        obj_repr = {}
        obj_repr["name"] = self.__class__.__name__
        obj_repr["len"] = len(self)
        obj_repr["batch_size_train_val"] = self.batch_size_train_val
        obj_repr["batch_size_test"] = self.batch_size_test
        obj_repr["num_node_feats"] = self.num_node_feats
        obj_repr["num_node_feats_with_positional_encoding"] = self.num_node_feats_with_positional_encoding
        obj_repr["num_edge_feats"] = self.num_edge_feats
        obj_repr["num_labels"] = len(self.label_cols)
        obj_repr["collate_fn"] = self.collate_fn.__name__
        obj_repr["featurization"] = self.featurization
        return obj_repr

    def __repr__(self):
        """Controls how the class is printed"""
        return omegaconf.OmegaConf.to_yaml(self.to_dict())


class DGLOGBDataModule(DGLFromSmilesDataModule):
    """Load an OGB GraphProp dataset."""

    def __init__(
        self,
        dataset_name: str,
        cache_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        weights_col: str = None,
        weights_type: str = None,
        batch_size_train_val: int = 16,
        batch_size_test: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        collate_fn: Optional[Callable] = None,
    ):
        """

        Parameters:
            dataset_name: Name of the OGB dataset to load. Examples of possible datasets are
                "ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molfreesolv".
            cache_data_path: path where to save or reload the cached data. The path can be
                remote (S3, GS, etc).
            featurization: args to apply to the SMILES to DGL featurizer.
            batch_size_train_val: batch size for training and val dataset.
            batch_size_test: batch size for test dataset.
            num_workers: Number of workers for the dataloader. Use -1 to use all available
                cores.
            pin_memory: Whether to pin on paginated CPU memory for the dataloader.
            featurization_n_jobs: Number of cores to use for the featurization.
            featurization_progress: whether to show a progress bar during featurization.
            collate_fn: A custom torch collate function. Default is to `goli.data.goli_collate_fn`
        """

        self.dataset_name = dataset_name

        # Get OGB metadata
        self.metadata = self._get_dataset_metadata(self.dataset_name)

        # Get dataset
        df, idx_col, smiles_col, label_cols, splits_path = self._load_dataset(self.metadata)

        # Config for datamodule
        dm_args = {}
        dm_args["df"] = df
        dm_args["cache_data_path"] = cache_data_path
        dm_args["featurization"] = featurization
        dm_args["smiles_col"] = smiles_col
        dm_args["label_cols"] = label_cols
        dm_args["idx_col"] = idx_col
        dm_args["splits_path"] = splits_path
        dm_args["batch_size_train_val"] = batch_size_train_val
        dm_args["batch_size_test"] = batch_size_test
        dm_args["num_workers"] = num_workers
        dm_args["pin_memory"] = pin_memory
        dm_args["featurization_n_jobs"] = featurization_n_jobs
        dm_args["featurization_progress"] = featurization_progress
        dm_args["persistent_workers"] = persistent_workers
        dm_args["collate_fn"] = collate_fn
        dm_args["weights_col"] = weights_col
        dm_args["weights_type"] = weights_type

        # Init DGLFromSmilesDataModule
        super().__init__(**dm_args)

    def to_dict(self):
        obj_repr = {}
        obj_repr["dataset_name"] = self.dataset_name
        obj_repr.update(super().to_dict())
        return obj_repr

    # Private methods

    def _load_dataset(self, metadata: dict):
        """Download, extract and load an OGB dataset."""

        base_dir = fs.get_cache_dir("ogb")
        dataset_dir = base_dir / metadata["download_name"]

        if not dataset_dir.exists():

            # Create cache filepath for zip file and associated folder
            dataset_path = base_dir / f"{metadata['download_name']}.zip"

            # Download it
            if not dataset_path.exists():
                logger.info(f"Downloading {metadata['url']} to {dataset_path}")
                fs.copy(metadata["url"], dataset_path, progress=True)

            # Extract
            zf = zipfile.ZipFile(dataset_path)
            zf.extractall(base_dir)

        # Load CSV file
        df_path = dataset_dir / "mapping" / "mol.csv.gz"
        logger.info(f"Loading {df_path} in memory.")
        df = pd.read_csv(df_path)

        # Load split from the OGB dataset and save them in a single CSV file
        split_name = metadata["split"]
        train_split = pd.read_csv(dataset_dir / "split" / split_name / "train.csv.gz", header=None)  # type: ignore
        val_split = pd.read_csv(dataset_dir / "split" / split_name / "valid.csv.gz", header=None)  # type: ignore
        test_split = pd.read_csv(dataset_dir / "split" / split_name / "test.csv.gz", header=None)  # type: ignore

        splits = pd.concat([train_split, val_split, test_split], axis=1)  # type: ignore
        splits.columns = ["train", "val", "test"]

        splits_path = dataset_dir / "split" / f"{split_name}.csv.gz"
        logger.info(f"Saving splits to {splits_path}")
        splits.to_csv(splits_path, index=None)

        # Get column names: OGB columns are predictable
        idx_col = df.columns[-1]
        smiles_col = df.columns[-2]
        label_cols = df.columns[:-2].to_list()

        return df, idx_col, smiles_col, label_cols, splits_path

    def _get_dataset_metadata(self, dataset_name: str):
        ogb_metadata = self._get_ogb_metadata()

        if dataset_name not in ogb_metadata.index:
            raise ValueError(f"'{dataset_name}' is not a valid dataset name.")

        return ogb_metadata.loc[dataset_name].to_dict()

    def _get_ogb_metadata(self):
        """Get the metadata of OGB GraphProp datasets."""

        with importlib.resources.open_text("ogb.graphproppred", "master.csv") as f:
            ogb_metadata = pd.read_csv(f)

        ogb_metadata = ogb_metadata.set_index(ogb_metadata.columns[0])
        ogb_metadata = ogb_metadata.T

        # Only keep datasets of type 'mol'
        ogb_metadata = ogb_metadata[ogb_metadata["data type"] == "mol"]

        return ogb_metadata
