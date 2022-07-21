from typing import Type, List, Dict, Union, Any, Callable, Optional, Tuple, Iterable

import os
from functools import partial
import importlib.resources
import zipfile
from copy import deepcopy

from loguru import logger
import fsspec
import omegaconf
from tqdm import tqdm
from joblib import Parallel, delayed
import tempfile

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import dgl
import pytorch_lightning as pl

from goli.utils import fs
from goli.features import mol_to_graph_dict, mol_to_dglgraph_signature, mol_to_dglgraph, GraphDict, mol_to_pyggraph
from goli.data.collate import goli_collate_fn
from goli.utils.arg_checker import check_arg_iterator

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import Subset

import datamol as dm
#from pprint import pprint

PCQM4M_meta = {
    "num tasks": 1,
    "eval metric": "mae",
    "download_name": "pcqm4m_kddcup2021",
    "url": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip",
    "data type": "mol",
    "has_node_attr": True,
    "has_edge_attr": True,
    "task type": "regression",
    "num classes": -1,
    "split": "scaffold",
    "additional node files": "None",
    "additional edge files": "None",
    "binary": False,
    "version": 1,
}

PCQM4Mv2_meta = deepcopy(PCQM4M_meta)
PCQM4Mv2_meta.update(
    {
        "download_name": "pcqm4m-v2",
        "url": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip",
        "version": 2,
    }
)


@staticmethod
def smiles_to_unique_mol_ids(smiles: List[str]):
    """This function takes a list of smiles and finds the corresponding datamol unique_id in an element-wise fashion, returning the corresponding unique_ids."""
    unique_mol_ids = []
    for smile in smiles:
        mol = dm.to_mol(mol=smile)
        id = dm.unique_id(mol)
        unique_mol_ids.append(id)
    return unique_mol_ids


@staticmethod
def get_indices(data_list: List, query_list: List):
    """This function searches for data_list's elements within query_list and returns the indices of query_list where they were found."""
    not_found = set(query_list)
    datum_idx_map = {}
    for index, datum in enumerate(data_list):
        if datum in not_found:
            datum_idx_map[datum] = index
            not_found.remove(datum)
            if len(not_found) == 0:
                break
    return [datum_idx_map[d] for d in query_list]


# Modify this if the dictionaries change structure.
@staticmethod
def concatenate_respective_lists_and_save_indices(
    dict_containing_lists: Dict[str, List], key_containing_list: str
):
    """Given a dict with values List, this function concatenates all the lists into one, and saves the respective indices of each key's list in the concatenated list.

    Currently this function assumes the following structure:
        {
            "key": {
                "key_containing_list": []
                "other_key": Any
                ...
            }
            ...
        }
    """
    concatenated_list = []
    dict_respective_indices = {}
    start = 0
    for key, value in dict_containing_lists.items():
        list = value[key_containing_list]
        concatenated_list.extend(list)

        end = start + len(list) - 1

        dict_respective_indices[key] = {}
        dict_respective_indices[key]["start"] = start
        dict_respective_indices[key]["end"] = end

        start = end + 1
    return concatenated_list, dict_respective_indices


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

class SingleTaskDataset(Dataset):
    def __init__(
        self,
        labels: Union[torch.Tensor, np.ndarray],
        features: Optional[List[dgl.DGLGraph]] = None,
        smiles: Optional[List[str]] = None,
        indices: Optional[List[str]] = None,
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        self.features = features
        self.labels = labels
        self.smiles = smiles
        self.indices = indices
        self.weights = weights

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        datum = {}

        if self.features is not None:
            datum["features"] = self.features[idx]

        if self.labels is not None:
            datum["labels"] = self.labels[idx]

        if self.smiles is not None:
            datum["smiles"] = self.smiles[idx]

        if self.indices is not None:
            datum["indices"] = self.indices[idx]

        if self.weights is not None:
            datum["weights"] = self.weights[idx]

        return datum

class MultitaskDGLDataset(Dataset):
    """This class holds the information for the multitask dataset.

    Several single-task datasets can be merged to create a multi-task dataset."""
    def __init__(self, datasets: Dict[str, SingleTaskDataset]):
        super().__init__()
        self.datasets = datasets

        task = next(iter(self.datasets))
        if "features" in datasets[task].__getitem__(0):
            self.mol_ids, self.smiles, self.labels, self.features = self.merge(self.datasets)
        else:
            self.mol_ids, self.smiles, self.labels = self.merge(self.datasets)
        #self.labels_size = self.set_label_size_dict()

    def __len__(self):
        return len(self.mol_ids)

    def __getitem__(self, idx):
        datum = {}

        if self.mol_ids is not None:
            datum["mol_ids"] = self.mol_ids[idx]

        if self.smiles is not None:
            datum["smiles"] = self.smiles[idx]

        if self.labels is not None:
            datum["labels"] = self.labels[idx]

        if self.features is not None:
            datum["features"] = self.features[idx]

        return datum

    def merge(self, datasets):
        all_smiles = []
        all_features = []
        all_labels = []

        all_tasks = []
        for task, ds in datasets.items():
            # Get data from single task ds
            ds_smiles = [ds.__getitem__(i)["smiles"] for i in range(len(ds))]
            ds_labels = [ds.__getitem__(i)["labels"] for i in range(len(ds))]
            if "features" in ds.__getitem__(0):
                ds_features = [ds.__getitem__(i)["features"] for i in range(len(ds))]
            else:
                ds_features = None

            all_smiles.extend(ds_smiles)
            all_labels.extend(ds_labels)
            if ds_features is not None: all_features.extend(ds_features)

            task_list = [task] * ds.__len__()
            all_tasks.extend(task_list)

        mol_ids = []
        # Get all unique mol ids.
        all_mol_ids = smiles_to_unique_mol_ids(all_smiles)
        unique_mol_ids, inv = np.unique(all_mol_ids, return_inverse=True)
        mol_ids = unique_mol_ids

        #print("\n\n\nUnique mol ID's! ")
        #pprint(mol_ids)
        #print("\nAll mol IDs ")
        #pprint(all_mol_ids)
        #print("\nAll smiles ")
        #pprint(all_smiles)

        # Store the smiles.
        #smiles = [[] for i in range(len(mol_ids))]
        smiles = [[]] * len(mol_ids)
        for all_idx, unique_idx in enumerate(inv):
            smiles[unique_idx].append(all_smiles[all_idx])

        # Store the labels.
        #labels = [{}] * len(mol_ids)
        labels = [{} for i in range(len(mol_ids))]
        for all_idx, unique_idx in enumerate(inv):
            task = all_tasks[all_idx]
            label = all_labels[all_idx]
            labels[unique_idx][task] = label

        #print("All features", all_features)
        #print("length of mol_id", len(mol_ids))
        #print("length of all features", len(all_features))

        # Store the features
        if len(all_features) > 0:
            features = [-1 for i in range(len(mol_ids))]
            for all_idx, unique_idx in enumerate(inv):
                features[unique_idx] = all_features[all_idx]
            return mol_ids, smiles, labels, features
        else:
            return mol_ids, smiles, labels

    def set_label_size_dict(self):
        # This gives the number of labels to predict for a given task.
        task_labels_size = {}
        for task, ds in self.datasets.items():
            label = ds.__getitem__(0)["labels"]       # Assume for a fixed task, the label dimension is the same across data points, so we can choose the first data point for simplicity.
            torch_label = torch.from_numpy(label)
            #torch_label = label
            #print("\n\n\nTorch label: ", torch_label)
            #print("Type: ", type(torch_label))
            #print("The size: ", torch_label.size())
            task_labels_size[task] = torch_label.size()
        return task_labels_size

    # def print_data(self):
    #     print("\n\n mol_ids")
    #     pprint(self.mol_ids)
    #     print("\n\n smiles")
    #     pprint(self.smiles)
    #     print("\n\n labels")
    #     pprint(self.labels)

class BaseDataModule(pl.LightningDataModule):
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
            # Some values become `inf` when changing data type. `mask_nan` deals with that
            self.collate_fn = partial(goli_collate_fn, mask_nan=0)
            self.collate_fn.__name__ = goli_collate_fn.__name__
        else:
            self.collate_fn = collate_fn

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._predict_ds = None

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

    def predict_dataloader(self, **kwargs):

        return self._dataloader(
            dataset=self.predict_ds,  # type: ignore
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

    @property
    def predict_ds(self):
        """Get the dataset used for the prediction"""
        if self._predict_ds is None:
            return self.test_ds
        else:
            return self._predict_ds

    @predict_ds.setter
    def predict_ds(self, value):
        """Set the dataset for the prediction"""
        self._predict_ds = value

    def get_first_graph(self):
        raise NotImplementedError()

    # Private methods

    @staticmethod
    def _read_csv(path, **kwargs):
        if str(path).endswith((".csv", ".csv.gz", ".csv.zip", ".csv.bz2")):
            sep = ","
        elif str(path).endswith((".tsv", ".tsv.gz", ".tsv.zip", ".tsv.bz2")):
            sep = "\t"
        else:
            raise ValueError(f"unsupported file `{path}`")
        kwargs.setdefault("sep", sep)
        df = pd.read_csv(path, **kwargs)
        return df

    def _dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool):
        """Get a dataloader for a given dataset"""

        if self.num_workers == -1:
            num_workers = os.cpu_count()
            num_workers = num_workers if num_workers is not None else 0
        else:
            num_workers = self.num_workers

        # TODO (Andy): if self.on_ipu, use poptorch.DataLoader instead of torch.DataLoader
        # To change how the padding is done, check the `goli_collate_fn` or `self.collate_fn`

        # TODO (Gabriela): Develop new dataloader to handle special batching.
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

class DGLFromSmilesDataModule(BaseDataModule):
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
        sample_size: Union[int, float, Type[None]] = None,
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
        featurization_backend: str = "loky",
        collate_fn: Optional[Callable] = None,
        prepare_dict_or_graph: str = "dgldict",
        dataset_class: type = DGLDataset,
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
            label_cols: Name of the columns to use as labels, with different options.

                - `list`: A list of all column names to use
                - `None`: All the columns are used except the SMILES one.
                - `str`: The name of the single column to use
                - `*str`: A string starting by a `*` means all columns whose name
                  ends with the specified `str`
                - `str*`: A string ending by a `*` means all columns whose name
                  starts with the specified `str`

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
            featurization_backend: The backend to use for the molecular featurization.

                - "multiprocessing": Found to cause less memory issues.
                - "loky": joblib's Default. Found to cause memory leaks.
                - "threading": Found to be slow.

            collate_fn: A custom torch collate function. Default is to `goli.data.goli_collate_fn`
            sample_size:

                - `int`: The maximum number of elements to take from the dataset.
                - `float`: Value between 0 and 1 representing the fraction of the dataset to consider
                - `None`: all elements are considered.
            prepare_dict_or_graph: Whether to preprocess all molecules as DGL graphs or dict.
                Possible options:

                - "graph": Process molecules as dgl.DGLGraph. It's slower during pre-processing
                  and requires more RAM. It is faster during training with `num_workers=0`, but
                  slower with larger `num_workers`.
                - "dict": Process molecules as a Dict. It's faster and requires less RAM during
                  pre-processing. It is slower during training with with `num_workers=0` since
                  DGLGraphs will be created during data-loading, but faster with large
                  `num_workers`, and less likely to cause memory issues with the parallelization.
            dataset_class: The class used to create the dataset from which to sample.
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
        self.featurization = featurization if featurization is not None else {}

        self.smiles_col = smiles_col
        self.label_cols = self._parse_label_cols(label_cols, smiles_col)
        self.idx_col = idx_col
        self.sample_size = sample_size

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
        self.featurization_backend = featurization_backend

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._predict_ds = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.dataset_class = dataset_class

        # Depreciated options
        if prepare_dict_or_graph == "dict":
            logger.warning("Depreciated: Use `prepare_dict_or_graph = 'dgldict'` instead of 'dict'")
            prepare_dict_or_graph = "dgldict"
        elif prepare_dict_or_graph == "graph":
            logger.warning("Depreciated: Use `prepare_dict_or_graph = 'dglgraph'` instead of 'graph'")
            prepare_dict_or_graph = "dglgraph"

        # Whether to transform the smiles into a dglgraph or a dictionary compatible with dgl
        if prepare_dict_or_graph == "dgldict":
            self.smiles_transformer = partial(mol_to_dglgraph_dict, **featurization)
        elif prepare_dict_or_graph == "dglgraph":
            self.smiles_transformer = partial(mol_to_dglgraph, **featurization)
        else:
            raise ValueError(
                f"`prepare_dict_or_graph` should be either 'dgldict' or 'dglgraph', Provided: `{prepare_dict_or_graph}`"
            )

    def prepare_data(self): # Can create train_ds, val_ds and test_ds here instead of setup. Be careful with the featurization (to not compute several times).
        """Called only from a single process in distributed settings. Steps:

        - If cache is set and exists, reload from cache.
        - Load the dataframe if its a path.
        - Extract smiles and labels from the dataframe.
        - Compute the features.
        - Compute or set split indices.
        - Make the list of dict dataset.
        """

        if self._load_from_cache():
            return True

        if self.df is None:
            # Only load the useful columns, as some dataset can be very large
            # when loading all columns
            label_cols = check_arg_iterator(self.label_cols, enforce_type=list)
            usecols = (
                check_arg_iterator(self.smiles_col, enforce_type=list)
                + label_cols
                + check_arg_iterator(self.idx_col, enforce_type=list)
                + check_arg_iterator(self.weights_col, enforce_type=list) # Can remove the weights (Dom says it's OK)
            )
            label_dtype = {col: np.float16 for col in label_cols}

            df = self._read_csv(self.df_path, usecols=usecols, dtype=label_dtype)
        else:
            df = self.df

        df = self._sub_sample_df(df)

        logger.info(f"Prepare dataset with {len(df)} data points.")

        # Extract smiles and labels
        smiles, labels, sample_idx, extras = self._extract_smiles_labels(
            df,
            smiles_col=self.smiles_col,
            label_cols=self.label_cols,
            idx_col=self.idx_col,
            weights_col=self.weights_col,
            weights_type=self.weights_type,
        )

        # Convert SMILES to features (graphs, fingerprints, etc.)
        features, idx_none = self._featurize_molecules(smiles, sample_idx)

        # Filter the molecules, labels, etc. for the molecules that failed featurization
        df, features, smiles, labels, sample_idx, extras = self._filter_none_molecules(
            idx_none, df, features, smiles, labels, sample_idx, extras
        )

        # Get splits indices
        self.train_indices, self.val_indices, self.test_indices = self._get_split_indices(
            len(df),
            split_val=self.split_val,
            split_test=self.split_test,
            split_seed=self.split_seed,
            splits_path=self.splits_path,
            sample_idx=sample_idx,
        )

        # Make the torch datasets (mostly a wrapper there is no memory overhead here)
        self.dataset = self.dataset_class(
            smiles=smiles,
            features=features,
            labels=labels,
            **extras,
        )

        self._save_to_cache()

    def setup(self, stage: str = None): # Can get rid of setup because a single dataset will have molecules exclusively in train, val or test
        """Prepare the torch dataset. Called on every GPUs. Setting state here is ok."""

        if stage == "fit" or stage is None:
            self.train_ds = Subset(self.dataset, self.train_indices)  # type: ignore
            self.val_ds = Subset(self.dataset, self.val_indices)  # type: ignore

        if stage == "test" or stage is None:
            self.test_ds = Subset(self.dataset, self.test_indices)  # type: ignore

    def _featurize_molecules(
        self, smiles: Iterable[str], sample_idx: Iterable[int] = None
    ) -> Tuple[List, List]:
        """
        Precompute the features (graphs, fingerprints, etc.) from the SMILES.
        Features are computed from `self.smiles_transformer`.
        A warning is issued to mention which molecules failed featurization.

        Note:
            (hadim): in case of very large dataset we could:
            - or cache the data and read from it during `next(iter(dataloader))`
            - or compute the features on-the-fly during `next(iter(dataloader))`
            For now we compute in advance and hold everything in memory.

        Parameters:
            smiles: A list of all the molecular SMILES to featurize
            sample_idx: The indexes corresponding to the sampled SMILES.
                If not provided, computed from `numpy.arange`.

        Returns:
            features: A list of all the featurized molecules
            idx_none: A list of the indexes that failed featurization
        """

        # Loop all the smiles and compute the features
        if self.featurization_n_jobs == 0:
            features = [self.smiles_transformer(s) for s in tqdm(smiles)]
        else:
            features = Parallel(n_jobs=self.featurization_n_jobs, backend=self.featurization_backend)(
                delayed(self.smiles_transformer)(s) for s in tqdm(smiles)
            )

        # Warn about None molecules
        if sample_idx is None:
            sample_idx = np.arange(len(smiles))
        idx_none = [ii for ii, feat in enumerate(features) if feat is None]
        if len(idx_none) > 0:
            mols_to_msg = [f"{sample_idx[idx]}: {smiles[idx]}" for idx in idx_none]
            msg = "\n".join(mols_to_msg)
            logger.warning(
                (f"{len(idx_none)} molecules will be removed since they failed featurization:\n" + msg)
            )

        return features, idx_none

    @staticmethod
    def _filter_none_molecules(
        idx_none: Iterable,
        *args: Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor, list, tuple, Dict[Any, Iterable]],
    ) -> List[Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor, list, tuple, Dict[Any, Iterable]]]:
        """
        Filter the molecules, labels, etc. for the molecules that failed featurization.

        Parameters:
            idx_none: A list of the indexes that failed featurization
            args: Any argument from which to filter the failed SMILES.
                Can be a `list`, `tuple`, `Tensor`, `np.array`, `Dict`, `pd.DataFrame`, `pd.Series`.
                Otherwise, it is not filtered.
                WARNING: If a `pd.DataFrame` or `pd.Series` is passed, it filters by the row indexes,
                NOT by the `DataFrame.index` or `Series.index`! Be careful!

        Returns:
            out: All the `args` with the indexes from `idx_none` removed.
        """
        if len(idx_none) == 0:
            return args
        idx_none = np.asarray(idx_none)

        out = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                new = arg.drop(arg.index[idx_none], axis=0)
            elif isinstance(arg, pd.Series):
                new = arg.drop(arg.index[idx_none], axis=0)
            elif isinstance(arg, np.ndarray):
                new = np.delete(arg, idx_none, axis=0)
            elif isinstance(arg, torch.Tensor):
                not_none = torch.ones(arg.shape[0], dtype=bool)
                not_none[idx_none] = False
                new = arg[not_none]
            elif isinstance(arg, (list, tuple)):
                arg = list(arg)
                new = [elem for ii, elem in enumerate(arg) if ii not in idx_none]
            elif isinstance(arg, dict):
                new = {}
                for key, val in arg.items():
                    new[key] = DGLFromSmilesDataModule._filter_none_molecules(idx_none, val)
            else:
                new = arg
            out.append(new)

        out = tuple(out) if len(out) > 1 else out[0]

        return out

    def _parse_label_cols(self, label_cols: Union[Type[None], str, List[str]], smiles_col: str) -> List[str]:
        r"""
        Parse the choice of label columns depending on the type of input.
        The input parameters `label_cols` and `smiles_col` are described in
        the `__init__` method.
        """
        if self.df is None:
            # Only load the useful columns, as some dataset can be very large
            # when loading all columns
            df = self._read_csv(self.df_path, nrows=0)
        else:
            df = self.df
        cols = list(df.columns)

        # A star `*` at the beginning or end of the string specifies to look for all
        # columns that starts/end with a specific string
        if isinstance(label_cols, str):
            if label_cols[0] == "*":
                label_cols = [col for col in cols if str(col).endswith(label_cols[1:])]
            elif label_cols[-1] == "*":
                label_cols = [col for col in cols if str(col).startswith(label_cols[:-1])]
            else:
                label_cols = [label_cols]

        elif label_cols is None:
            label_cols = [col for col in cols if col != smiles_col]

        return check_arg_iterator(label_cols, enforce_type=list)

    @property
    def is_prepared(self):
        if not hasattr(self, "dataset"):
            return False
        return getattr(self, "dataset") is not None

    @property
    def is_setup(self):
        if not (hasattr(self, "train_ds") or hasattr(self, "test_ds")):
            return False
        return (getattr(self, "train_ds") is not None) or (getattr(self, "test_ds") is not None)

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
        if "edge_feat" in graph.edata.keys():
            return graph.edata["edge_feat"].shape[1]  # type: ignore
        else:
            return 0

    def get_first_graph(self):
        """
        Low memory footprint method to get the first datapoint DGL graph.
        The first 10 rows of the data are read in case the first one has a featurization
        error. If all 10 first element, then `None` is returned, otherwise the first
        graph to not fail is returned.
        """
        if self.df is None:
            df = self._read_csv(self.df_path, nrows=10)
        else:
            df = self.df.iloc[0:10, :]

        smiles, _, _, _ = self._extract_smiles_labels(df, self.smiles_col, self.label_cols)

        graph = None
        for s in smiles:
            graph = self.smiles_transformer(s)
            if graph is not None:
                break

        if isinstance(graph, GraphDict):
            graph = graph.make_dgl_graph(mask_nan=0.0)

        return graph

    # Private methods

    def _save_to_cache(self):
        """Save the built dataset, indices and featurization arguments into a cache file."""

        # Cache on disk
        if self.cache_data_path is not None:
            logger.info(f"Write prepared datamodule to {self.cache_data_path}")
            cache = {}
            cache["dataset"] = self.dataset
            cache["train_indices"] = self.train_indices
            cache["val_indices"] = self.val_indices
            cache["test_indices"] = self.test_indices

            # Save featurization args used
            cache["featurization_args"] = mol_to_dglgraph_signature(dict(self.featurization or {}))

            with fsspec.open(self.cache_data_path, "wb", compression="infer") as f:
                torch.save(cache, f)

    def _load_from_cache(self):
        """Attempt to reload the data from cache. Return True if data has been
        reloaded from the cache.
        """

        if self.cache_data_path is None:
            # Cache path is not provided.
            return False

        if not fs.exists(self.cache_data_path):
            # Cache patch does not exist.
            return False

        # Reload from cache if it exists and is valid
        logger.info(f"Try reloading the data module from {self.cache_data_path}.")

        # Load cache and save it locally in a temp folder.
        # This allows loading the cache much faster if it is zipped or in the cloud
        filesystem, _ = fsspec.core.url_to_fs(self.cache_data_path, mode="rb")
        protocol = check_arg_iterator(filesystem.protocol, enforce_type=list)
        filesystem = fsspec.filesystem(
            "filecache",
            target_protocol=protocol[0],
            target_options={"anon": True},
            cache_storage=tempfile.TemporaryDirectory().name,
            compression="infer",
        )
        with filesystem.open(self.cache_data_path, "rb", compression="infer") as f:
            cache = torch.load(f)

        # Are the required keys present?
        excepted_cache_keys = {
            "dataset",
            "test_indices",
            "train_indices",
            "val_indices",
        }
        if not set(cache.keys()) != excepted_cache_keys:
            logger.info(
                f"Cache looks invalid with keys: {cache.keys()}. Excepted keys are {excepted_cache_keys}"
            )
            logger.info("Fallback to regular data preparation steps.")
            return False

        # Is the featurization signature the same?
        current_signature = mol_to_dglgraph_signature(dict(self.featurization or {}))
        cache_signature = mol_to_dglgraph_signature(cache["featurization_args"])

        if current_signature != cache_signature:
            logger.info(f"Cache featurizer arguments are different than the provided ones.")
            logger.info(f"Cache featurizer arguments: {cache_signature}")
            logger.info(f"Provided featurizer arguments: {current_signature}.")
            logger.info("Fallback to regular data preparation steps.")
            return False

        # At this point the cache can be loaded

        self.dataset = cache["dataset"]
        self.train_indices = cache["train_indices"]
        self.val_indices = cache["val_indices"]
        self.test_indices = cache["test_indices"]

        logger.info(f"Datamodule correctly reloaded from cache.")

        return True

    def _extract_smiles_labels(
        self,
        df: pd.DataFrame,
        smiles_col: str = None,
        label_cols: List[str] = [],
        idx_col: str = None,
        weights_col: str = None,
        weights_type: str = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, Union[Type[None], np.ndarray], Dict[str, Union[Type[None], np.ndarray]]
    ]:
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

        label_cols = check_arg_iterator(label_cols, enforce_type=list)

        smiles = df[smiles_col].values
        if len(label_cols) > 0:
            labels = [pd.to_numeric(df[col], errors="coerce") for col in label_cols]
            labels = np.stack(labels, axis=1)
        else:
            labels = np.zeros([len(smiles), 0])

        indices = None
        if idx_col is not None:
            indices = df[idx_col].values

        sample_idx = df.index.values

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
                weights[labels == 1] = ratio_pos_neg**-1

            elif weights_type == "sample_balanced":
                ratio_pos_neg = np.sum(labels, axis=0, keepdims=1) / labels.shape[0]
                weights = np.zeros(labels.shape)
                weights[labels == 0] = ratio_pos_neg
                weights[labels == 1] = ratio_pos_neg**-1
                weights = np.prod(weights, axis=1)

            else:
                raise ValueError(f"Undefined `weights_type` {weights_type}")

            weights /= np.max(weights)  # Put the max weight to 1

        extras = {"indices": indices, "weights": weights}
        return smiles, labels, sample_idx, extras

    def _get_split_indices(
        self,
        dataset_size: int,
        split_val: float,
        split_test: float,
        sample_idx: Optional[Iterable[int]] = None,
        split_seed: int = None,
        splits_path: Union[str, os.PathLike] = None,
    ):
        """Compute indices of random splits."""

        if sample_idx is None:
            sample_idx = np.arange(dataset_size)

        if splits_path is None:
            # Random splitting
            train_indices, val_test_indices = train_test_split(
                sample_idx,
                test_size=split_val + split_test,
                random_state=split_seed,
            )

            sub_split_test = split_test / (split_test + split_val)
            if split_test > 0:
                val_indices, test_indices = train_test_split(
                    val_test_indices,
                    test_size=sub_split_test,
                    random_state=split_seed,
                )
            else:
                val_indices = val_test_indices
                test_indices = np.array([])

        else:
            # Split from an indices file
            with fsspec.open(str(splits_path)) as f:
                splits = self._read_csv(splits_path)

            train_indices = splits["train"].dropna().astype("int").tolist()
            val_indices = splits["val"].dropna().astype("int").tolist()
            test_indices = splits["test"].dropna().astype("int").tolist()

        # Filter train, val and test indices
        _, train_idx, _ = np.intersect1d(sample_idx, train_indices, return_indices=True)
        train_indices = train_idx.tolist()
        _, valid_idx, _ = np.intersect1d(sample_idx, val_indices, return_indices=True)
        val_indices = valid_idx.tolist()
        _, test_idx, _ = np.intersect1d(sample_idx, test_indices, return_indices=True)
        test_indices = test_idx.tolist()

        return train_indices, val_indices, test_indices

    def _sub_sample_df(self, df):
        # Sub-sample the dataframe
        if isinstance(self.sample_size, int):
            n = min(self.sample_size, df.shape[0])
            df = df.sample(n=n)
        elif isinstance(self.sample_size, float):
            df = df.sample(f=self.sample_size)
        elif self.sample_size is None:
            pass
        else:
            raise ValueError(f"Wrong value for `self.sample_size`: {self.sample_size}")

        return df

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current DataModule
        """
        if self.df is None:
            df = self._read_csv(self.df_path, usecols=[self.smiles_col])
        else:
            df = self.df

        return len(df)

    def to_dict(self):
        obj_repr = {}
        obj_repr["name"] = self.__class__.__name__
        obj_repr["len"] = len(self)
        obj_repr["train_size"] = len(self.train_indices) if self.train_indices is not None else None
        obj_repr["val_size"] = len(self.val_indices) if self.val_indices is not None else None
        obj_repr["test_size"] = len(self.test_indices) if self.test_indices is not None else None
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

class DatasetParams():
   def __init__(
       self,
       task_name: str,
       df: pd.DataFrame = None,
       df_path: Optional[Union[str, os.PathLike]] = None,
       smiles_col: str = None,
       label_cols: List[str] = None,
       weights_col: str = None,
       weights_type: str = None,
       idx_col: str = None,
       sample_size: Union[int, float, Type[None]] = None,
       split_val: float = 0.2,
       split_test: float = 0.2,
       split_seed: int = None,
       splits_path: Optional[Union[str, os.PathLike]] = None,
   ):
       self.task_name = task_name
       self.df = df
       self.df_path = df_path
       self.smiles_col = smiles_col
       self.label_cols = label_cols
       self.weights_col = weights_col
       self.weights_type = weights_type
       self.idx_col = idx_col
       self.sample_size = sample_size
       self.split_val = split_val
       self.split_test = split_test
       self.split_seed = split_seed
       self.splits_path = splits_path
class MultitaskFromSmilesDataModule(BaseDataModule):
    def __init__(
        self,
        #task_df: Dict[str, pd.DataFrame] = None,
        #task_df_path: Optional[Dict[str, Union[str, os.PathLike]]] = None,
        #task_smiles_col: Dict[str, str] = None,
        #task_label_cols: Dict[str, List[str]] = None,
        #task_weights_col: Dict[str, str] = None,
        #task_weights_type: Dict[str, str] = None,
        #task_idx_col: Dict[str, str] = None,
        #task_sample_size: Dict[str, Union[int, float, None]] = None,
        #task_split_val: Dict[str, float] = None,
        #task_split_test: Dict[str, float]= None,
        #task_split_seed: Dict[str, int] = None,
        #task_splits_path: Optional[Dict[str, Union[str, os.PathLike]]] = None,
        task_specific_args: Dict[str, Any],         # Create a class that so that we can have a dictionary of class
        cache_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_train_val: int = 16,
        batch_size_test: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        featurization_backend: str = "loky",
        collate_fn: Optional[Callable] = None,
        prepare_dict_or_graph: str = "dgl_dict",
        dataset_class: type = MultitaskDGLDataset,
    ):
        """
        Parameters: only for parameters beginning with task_*, we have a dictionary where the key is the task name
        and the value is specified below.
            task_df: (value) a dataframe
            task_df_path: (value) a path to a dataframe to load (CSV file). `df` takes precedence over
                `df_path`.
            task_smiles_col: (value) Name of the SMILES column. If set to `None`, it will look for
                a column with the word "smile" (case insensitive) in it.
                If no such column is found, an error will be raised.
            task_label_cols: (value) Name of the columns to use as labels, with different options.

                - `list`: A list of all column names to use
                - `None`: All the columns are used except the SMILES one.
                - `str`: The name of the single column to use
                - `*str`: A string starting by a `*` means all columns whose name
                  ends with the specified `str`
                - `str*`: A string ending by a `*` means all columns whose name
                  starts with the specified `str`

            task_weights_col: (value) Name of the column to use as sample weights. If `None`, no
                weights are used. This parameter cannot be used together with `weights_type`.
            task_weights_type: (value) The type of weights to use. This parameter cannot be used together with `weights_col`.
                **It only supports multi-label binary classification.**

                Supported types:

                - `None`: No weights are used.
                - `"sample_balanced"`: A weight is assigned to each sample inversely
                    proportional to the number of positive value. If there are multiple
                    labels, the product of the weights is used.
                - `"sample_label_balanced"`: Similar to the `"sample_balanced"` weights,
                    but the weights are applied to each element individually, without
                    computing the product of the weights for a given sample.

            task_idx_col: (value) Name of the columns to use as indices. Unused if set to None.
            task_sample_size: (value)

                - `int`: The maximum number of elements to take from the dataset.
                - `float`: Value between 0 and 1 representing the fraction of the dataset to consider
                - `None`: all elements are considered.
            task_split_val: (value) Ratio for the validation split.
            task_split_test: (value) Ratio for the test split.
            task_split_seed: (value) Seed to use for the random split. More complex splitting strategy
                should be implemented.
            task_splits_path: (value) A path a CSV file containing indices for the splits. The file must contains
                3 columns "train", "val" and "test". It takes precedence over `split_val` and `split_test`.

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
            featurization_backend: The backend to use for the molecular featurization.

                - "multiprocessing": Found to cause less memory issues.
                - "loky": joblib's Default. Found to cause memory leaks.
                - "threading": Found to be slow.

            collate_fn: A custom torch collate function. Default is to `goli.data.goli_collate_fn`
            prepare_dict_or_graph: Whether to preprocess all molecules as DGL graphs or dict.
                Possible options:

                - "graph": Process molecules as dgl.DGLGraph. It's slower during pre-processing
                  and requires more RAM, but faster during training.
                - "dict": Process molecules as a Dict. It's faster and requires less RAM during
                  pre-processing, but slower during training since DGLGraphs will be created
                  during data-loading.
            dataset_class: The class used to create the dataset from which to sample.
        """
        super().__init__(
            batch_size_train_val=batch_size_train_val,
            batch_size_test=batch_size_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )

        #self.task_df = task_df
        #self.task_df_path = task_df_path
        #self.task_smiles_col = task_smiles_col
        #self.task_abel_cols = {task: self._parse_label_cols(task_df[task], task_df_path[task], task_label_cols[task], task_smiles_col[task]) for task in self.task_df.keys()} # Make sure this is not none
        #self.idx_col = task_idx_col
        #self.task_weights_col = task_weights_col      # Can remove this
        #self.task_weights_type = task_weights_type    # Can remove this
        #self.task_idx_col = task_idx_col
        #self.task_sample_size = task_sample_size
        #self.task_split_val = task_split_val
        #self.task_split_test = task_split_test
        #self.task_spit_seed = task_split_seed
        #self.task_splits_path = task_splits_path
        self.task_specific_args = task_specific_args

        self.featurization_n_jobs = featurization_n_jobs
        self.featurization_progress = featurization_progress
        self.featurization_backend = featurization_backend

        self.dataset_class = dataset_class

        self.task_train_indices = None
        self.task_val_indices = None
        self.task_test_indices = None

        self.single_task_datasets = None
        self.train_singletask_datasets = None
        self.val_singletask_datasets = None
        self.test_singletask_datasets = None

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        # Whether to transform the smiles into a dglgraph or a dictionary compatible with dgl
        if prepare_dict_or_graph == "dgl:dict":
            self.smiles_transformer = partial(mol_to_graph_dict, **featurization)
        elif prepare_dict_or_graph == "dgl:graph":
            self.smiles_transformer = partial(mol_to_dglgraph, **featurization)
        elif prepare_dict_or_graph == "pyg:graph":
            self.smiles_transformer = partial(mol_to_pyggraph, **featurization)
        else:
            raise ValueError(
                f"`prepare_dict_or_graph` should be either 'dgl:dict', 'dgl:graph' or 'pyg:graph', Provided: `{prepare_dict_or_graph}`"
            )

    def prepare_data(self):
        """Called only from a single process in distributed settings. Steps:

        - If each cache is set and exists, reload from cache and return. Otherwise,
        - For each single-task dataset:
            - Load its dataframe from a path (if provided)
            - Subsample the dataframe
            - Extract the smiles, labels from the dataframe
        - In the previous step, we were also able to get the unique smiles, which we use to compute the features
        - For each single-task dataframe and associated data (smiles, labels, etc.):
            - Filter out the data corresponding to molecules which failed featurization.
            - Create a corresponding SingletaskDataset
            - Split the SingletaskDataset according to the task-specific splits for train, val and test
        """
        # TODO (Gabriela): Implement the ability to load from cache.

        """Load all single-task dataframes."""
        task_df = {}
        for task, args in self.task_specific_args.items():
            if args["df"] is None:
                # Only load the useful columns, as some datasets can be very large when loading all columns.
                label_cols = check_arg_iterator(args["label_cols"], enforce_type=list)
                usecols = (
                    check_arg_iterator(args["smiles_col"], enforce_type=list)
                    + label_cols
                    + check_arg_iterator(args["idx_col"], enforce_type=list)
                    + check_arg_iterator(args["weights_col"], enforce_type=list)
                )
                label_dtype = {col: np.float16 for col in label_cols}

                task_df[task] = self._read_csv(args["df_path"], usecols=usecols, dtype=label_dtype)
            else:
                task_df[task] = args["df"]

        """Subsample the data frames and extract the necessary data to create SingleTaskDatasets for each task (smiles, labels, extras)."""
        task_dataset_args = {}
        for task in task_df.keys():
            task_dataset_args[task] = {}

        for task, df in task_df.items():
            # Subsample all the dataframes
            sample_size = self.task_specific_args[task]["sample_size"]
            df = self._sub_sample_df(df, sample_size)

            logger.info(f"Prepare single-task dataset for task '{task}' with {len(df)} data points.")

            # Extract smiles, labels, extras
            args = self.task_specific_args[task]
            smiles, labels, sample_idx, extras = self._extract_smiles_labels(
                df,
                smiles_col=args["smiles_col"],
                label_cols=args["label_cols"],
                idx_col=args["idx_col"],
                weights_col=args["weights_col"],
                weights_type=args["weights_type"],
            )

            # Store the relevant information for each task's dataset
            task_dataset_args[task]["smiles"] = smiles
            task_dataset_args[task]["labels"] = labels
            task_dataset_args[task]["sample_idx"] = sample_idx
            task_dataset_args[task]["extras"] = extras

        """Convert SMILES to features (graphs, fingerprints, etc.) for the unique molecules found."""
        all_smiles = []
        all_tasks = []
        for task, dataset_args in task_dataset_args.items():
            all_smiles.extend(dataset_args["smiles"])
            for count in range(len(dataset_args["smiles"])):
                all_tasks.append(task)
        # Get all unique mol ids
        all_mol_ids = smiles_to_unique_mol_ids(all_smiles)
        unique_mol_ids, inv = np.unique(all_mol_ids, return_inverse=True)
        mol_ids = unique_mol_ids
        # Get smiles to be used for featurization with those mol ids.
        smiles_to_featurize = []
        for id in range(len(mol_ids)):
            for all_idx, unique_idx in enumerate(inv):
                if unique_idx == id:  # If we found this unique mol id
                    smiles_to_featurize.append(all_smiles[all_idx])
                    break # No need to check if another smiles matches
        assert len(mol_ids) == len(smiles_to_featurize)

        # Convert SMILES to features
        features, idx_none = self._featurize_molecules(smiles_to_featurize) # sample_idx is removed ... might need to add it again later in another way

        # Store the features (including Nones, which will be filtered in the next step)
        for task in task_dataset_args.keys():
            task_dataset_args[task]["features"] = []
            task_dataset_args[task]["idx_none"] = []
        # Create a list of features matching up with the original smiles
        all_features = [features[unique_idx] for unique_idx in inv]

        # Add the features to the task-specific data
        for all_idx, task in enumerate(all_tasks):
            task_dataset_args[task]["features"].append(all_features[all_idx])
        # Update idx_none per task for later filtering
        for task, args in task_dataset_args.items():
            for idx, feat in enumerate(args["features"]):
                if feat == None: args["idx_none"].append(idx)

        """Filter data based on molecules which failed featurization. Create single task datasets as well."""
        self.single_task_datasets = {}
        for task, args in task_dataset_args.items():
            df, features, smiles, labels, sample_idx, extras = self._filter_none_molecules(
                args["idx_none"],
                task_df[task],
                args["features"],
                args["smiles"],
                args["labels"],
                args["sample_idx"],
                args["extras"]
            )

            # Update the data
            task_dataset_args[task]["smiles"] = smiles
            task_dataset_args[task]["labels"] = labels
            task_dataset_args[task]["features"] = features
            task_dataset_args[task]["sample_idx"] = sample_idx
            task_dataset_args[task]["extras"] = extras

            # We have the necessary components to create single-task datasets.
            self.single_task_datasets[task] = SingleTaskDataset(
                features=task_dataset_args[task]["features"],
                labels=task_dataset_args[task]["labels"],
                smiles=task_dataset_args[task]["smiles"],
                **task_dataset_args[task]["extras"],
            )

        """We split the data up to create train, val and test datasets"""
        self.task_train_indices = {}
        self.task_val_indices = {}
        self.task_test_indices = {}

        self.train_singletask_datasets = {}
        self.val_singletask_datasets = {}
        self.test_singletask_datasets = {}
        for task, df in task_df.items():
            train_indices, val_indices, test_indices = self._get_split_indices(
                len(df),
                split_val=self.task_specific_args[task]["split_val"],
                split_test=self.task_specific_args[task]["split_test"],
                split_seed=self.task_specific_args[task]["split_seed"],
                splits_path=self.task_specific_args[task]["splits_path"],
                sample_idx=task_dataset_args[task]["sample_idx"],
            )
            self.task_train_indices[task] = train_indices
            self.task_val_indices[task] = val_indices
            self.task_test_indices[task] = test_indices

            #train_singletask_datasets[task] = Subset()
            self.train_singletask_datasets[task] = Subset(self.single_task_datasets[task], train_indices)
            self.val_singletask_datasets[task] = Subset(self.single_task_datasets[task], val_indices)
            self.test_singletask_datasets[task] = Subset(self.single_task_datasets[task], test_indices)


        # TODO (Gabriela): Implement the ability to save to cache.

    def setup(self, stage: str = None): # Can possibly get rid of setup because a single dataset will have molecules exclusively in train, val or test
        """Prepare the torch dataset. Called on every GPUs. Setting state here is ok."""

        if stage == "fit" or stage is None:
            self.train_ds = MultitaskDGLDataset(self.train_singletask_datasets)  # type: ignore
            self.val_ds = MultitaskDGLDataset(self.val_singletask_datasets)  # type: ignore
            # Produce the label sizes
            train_label_sizes = self.train_ds.set_label_size_dict()
            val_label_sizes = self.val_ds.set_label_size_dict()
            #label_sizes = train_label_sizes
            #label_sizes.update(val_label_sizes)

        if stage == "test" or stage is None:
            self.test_ds = MultitaskDGLDataset(self.test_singletask_datasets)  # type: ignore
            # Produce the label sizes
            test_label_sizes = self.test_ds.set_label_size_dict()
            #label_sizes.update(test_label_sizes)

        # Produce the label sizes
        #train_label_sizes = self.train_ds.set_label_size_dict()
        #val_label_sizes = self.val_ds.set_label_size_dict()
        #test_label_sizes = self.test_ds.set_label_size_dict()

        #label_sizes = train_label_sizes
        #label_sizes.update(val_label_sizes)
        #label_sizes.update(test_label_sizes)

    # Cannot be used as is for the multitask version, because sample_idx does not apply.
    def _featurize_molecules(self, smiles: Iterable[str]) -> Tuple[List, List]:
        """
        Precompute the features (graphs, fingerprints, etc.) from the SMILES.
        Features are computed from `self.smiles_transformer`.
        A warning is issued to mention which molecules failed featurization.

        Note:
            (hadim): in case of very large dataset we could:
            - or cache the data and read from it during `next(iter(dataloader))`
            - or compute the features on-the-fly during `next(iter(dataloader))`
            For now we compute in advance and hold everything in memory.

        Parameters:
            smiles: A list of all the molecular SMILES to featurize
            sample_idx: The indexes corresponding to the sampled SMILES.
                If not provided, computed from `numpy.arange`.

        Returns:
            features: A list of all the featurized molecules
            idx_none: A list of the indexes that failed featurization
        """

        # Loop all the smiles and compute the features
        if self.featurization_n_jobs == 0:
            features = [self.smiles_transformer(s) for s in tqdm(smiles)]
        else:
            features = Parallel(n_jobs=self.featurization_n_jobs, backend=self.featurization_backend)(
                delayed(self.smiles_transformer)(s) for s in tqdm(smiles)
            )

        # Warn about None molecules
        idx_none = [ii for ii, feat in enumerate(features) if feat is None]
        if len(idx_none) > 0:
            mols_to_msg = [f"{smiles[idx]}" for idx in idx_none]
            msg = "\n".join(mols_to_msg)
            logger.warning(
                (f"{len(idx_none)} molecules will be removed since they failed featurization:\n" + msg)
            )

        return features, idx_none

    @staticmethod
    def _filter_none_molecules(
        idx_none: Iterable,
        *args: Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor, list, tuple, Dict[Any, Iterable]],
    ) -> List[Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor, list, tuple, Dict[Any, Iterable]]]:
        """
        Filter the molecules, labels, etc. for the molecules that failed featurization.

        Parameters:
            idx_none: A list of the indexes that failed featurization
            args: Any argument from which to filter the failed SMILES.
                Can be a `list`, `tuple`, `Tensor`, `np.array`, `Dict`, `pd.DataFrame`, `pd.Series`.
                Otherwise, it is not filtered.
                WARNING: If a `pd.DataFrame` or `pd.Series` is passed, it filters by the row indexes,
                NOT by the `DataFrame.index` or `Series.index`! Be careful!

        Returns:
            out: All the `args` with the indexes from `idx_none` removed.
        """
        if len(idx_none) == 0:
            return args
        idx_none = np.asarray(idx_none)

        out = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                new = arg.drop(arg.index[idx_none], axis=0)
            elif isinstance(arg, pd.Series):
                new = arg.drop(arg.index[idx_none], axis=0)
            elif isinstance(arg, np.ndarray):
                new = np.delete(arg, idx_none, axis=0)
            elif isinstance(arg, torch.Tensor):
                not_none = torch.ones(arg.shape[0], dtype=bool)
                not_none[idx_none] = False
                new = arg[not_none]
            elif isinstance(arg, (list, tuple)):
                arg = list(arg)
                new = [elem for ii, elem in enumerate(arg) if ii not in idx_none]
            elif isinstance(arg, dict):
                new = {}
                for key, val in arg.items():
                    new[key] = DGLFromSmilesDataModule._filter_none_molecules(idx_none, val)    # Careful
            else:
                new = arg
            out.append(new)

        out = tuple(out) if len(out) > 1 else out[0]

        return out

    def _parse_label_cols(
        self,
        df: pd.DataFrame,
        df_path: str,
        str: Optional[Union[str, os.PathLike]],
        label_cols: Union[Type[None], str, List[str]],
        smiles_col: str
        ) -> List[str]:
        r"""
        Parse the choice of label columns depending on the type of input.
        The input parameters `label_cols` and `smiles_col` are described in
        the `__init__` method.
        """
        if df is None:
            # Only load the useful columns, as some dataset can be very large
            # when loading all columns
            data_frame = self._read_csv(df_path, nrows=0)
        else:
            data_frame = df
        cols = list(data_frame.columns)

        # A star `*` at the beginning or end of the string specifies to look for all
        # columns that starts/end with a specific string
        if isinstance(label_cols, str):
            if label_cols[0] == "*":
                label_cols = [col for col in cols if str(col).endswith(label_cols[1:])]
            elif label_cols[-1] == "*":
                label_cols = [col for col in cols if str(col).startswith(label_cols[:-1])]
            else:
                label_cols = [label_cols]

        elif label_cols is None:
            label_cols = [col for col in cols if col != smiles_col]

        return check_arg_iterator(label_cols, enforce_type=list)

    @property
    def is_prepared(self):
        if not hasattr(self, "dataset"):
            return False
        return getattr(self, "dataset") is not None

    @property
    def is_setup(self):
        if not (hasattr(self, "train_ds") or hasattr(self, "test_ds")):
            return False
        return (getattr(self, "train_ds") is not None) or (getattr(self, "test_ds") is not None)

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
        if isinstance(graph, (dgl.DGLGraph, GraphDict)):
            graph = graph.ndata

        empty = torch.Tensor([])
        num_feats  = graph.get("feat", empty).shape[-1]
        num_feats += graph.get("pos_enc_feats_sign_flip", empty).shape[-1]
        num_feats += graph.get("pos_enc_feats_no_flip", empty).shape[-1]

        return num_feats

    @property
    def num_edge_feats(self):
        """Return the number of edge features in the first graph"""

        graph = self.get_first_graph()
        if isinstance(graph, (dgl.DGLGraph, GraphDict)):
            graph = graph.edata

        empty = torch.Tensor([])
        num_feats = graph.get("edge_feat", empty).shape[-1]

        return num_feats

    def get_first_graph(self):
        """
        Low memory footprint method to get the first datapoint DGL graph.
        The first 10 rows of the data are read in case the first one has a featurization
        error. If all 10 first element, then `None` is returned, otherwise the first
        graph to not fail is returned.
        """
        keys = list(self.task_specific_args.keys())
        task = keys[0]
        args = self.task_specific_args[task]
        if args["df"] is None:
            df = self._read_csv(args["df_path"], nrows=10)
        else:
            df = args["df"].iloc[0:10, :]

        smiles, labels, sample_idx, extras = self._extract_smiles_labels(
            df,
            smiles_col=args["smiles_col"],
            label_cols=args["label_cols"],
            idx_col=args["idx_col"],
            weights_col=args["weights_col"],
            weights_type=args["weights_type"],
        )

        #smiles, _, _, _ = self._extract_smiles_labels(df, self.smiles_col, self.label_cols)

        graph = None
        for s in smiles:
            graph = self.smiles_transformer(s, mask_nan=0.0)
            if graph is not None:
                break

        if isinstance(graph, GraphDict):
            graph = graph.make_dgl_graph(mask_nan=0.0)

        return graph

    ########################## Private methods ######################################
    def _save_to_cache(self):
        raise NotImplementedError()

    def _load_from_cache(self):
        raise NotImplementedError()

    def _extract_smiles_labels(
        self,
        df: pd.DataFrame,
        smiles_col: str = None,
        label_cols: List[str] = [],
        idx_col: str = None,
        weights_col: str = None,
        weights_type: str = None,
    ) -> Tuple[
        np.ndarray, np.ndarray, Union[Type[None], np.ndarray], Dict[str, Union[Type[None], np.ndarray]]
    ]:
        """For a given dataframe extract the SMILES and labels columns. Smiles is returned as a list
        of string while labels are returned as a 2D numpy array.
        """

        if smiles_col is None:      # Should we specify which dataset has caused the potential issue?
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

        label_cols = check_arg_iterator(label_cols, enforce_type=list)

        smiles = df[smiles_col].values
        if len(label_cols) > 0:
            labels = [pd.to_numeric(df[col], errors="coerce") for col in label_cols]
            labels = np.stack(labels, axis=1)
        else:
            labels = np.zeros([len(smiles), 0])

        indices = None                      # What are indices for?
        if idx_col is not None:
            indices = df[idx_col].values

        sample_idx = df.index.values

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
                weights[labels == 1] = ratio_pos_neg**-1

            elif weights_type == "sample_balanced":
                ratio_pos_neg = np.sum(labels, axis=0, keepdims=1) / labels.shape[0]
                weights = np.zeros(labels.shape)
                weights[labels == 0] = ratio_pos_neg
                weights[labels == 1] = ratio_pos_neg**-1
                weights = np.prod(weights, axis=1)

            else:
                raise ValueError(f"Undefined `weights_type` {weights_type}")

            weights /= np.max(weights)  # Put the max weight to 1

        extras = {"indices": indices, "weights": weights}
        return smiles, labels, sample_idx, extras

    def _get_split_indices(
        self,
        dataset_size: int,
        split_val: float,
        split_test: float,
        sample_idx: Optional[Iterable[int]] = None,
        split_seed: int = None,
        splits_path: Union[str, os.PathLike] = None,
    ):
        """Compute indices of random splits."""

        if sample_idx is None:
            sample_idx = np.arange(dataset_size)

        if splits_path is None:
            # Random splitting
            train_indices, val_test_indices = train_test_split(
                sample_idx,
                test_size=split_val + split_test,
                random_state=split_seed,
            )

            sub_split_test = split_test / (split_test + split_val)
            if split_test > 0:
                val_indices, test_indices = train_test_split(
                    val_test_indices,
                    test_size=sub_split_test,
                    random_state=split_seed,
                )
            else:
                val_indices = val_test_indices
                test_indices = np.array([])

        else:
            # Split from an indices file
            with fsspec.open(str(splits_path)) as f:
                splits = self._read_csv(splits_path)

            train_indices = splits["train"].dropna().astype("int").tolist()
            val_indices = splits["val"].dropna().astype("int").tolist()
            test_indices = splits["test"].dropna().astype("int").tolist()

        # Filter train, val and test indices
        _, train_idx, _ = np.intersect1d(sample_idx, train_indices, return_indices=True)
        train_indices = train_idx.tolist()
        _, valid_idx, _ = np.intersect1d(sample_idx, val_indices, return_indices=True)
        val_indices = valid_idx.tolist()
        _, test_idx, _ = np.intersect1d(sample_idx, test_indices, return_indices=True)
        test_indices = test_idx.tolist()

        return train_indices, val_indices, test_indices

    def _sub_sample_df(self, df, sample_size):
        # Sub-sample the dataframe
        if isinstance(sample_size, int):
            n = min(self.sample_size, df.shape[0])
            df = df.sample(n=n)
        elif isinstance(sample_size, float):
            df = df.sample(f=sample_size)
        elif sample_size is None:
            pass
        else:
            raise ValueError(f"Wrong value for `self.sample_size`: {self.sample_size}") # Maybe specify which task it was for?

        return df

    ########################## len  ###################################################
    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current DataModule
        """
        num_elements = 0
        for task, args in self.task_specific_args.items():
            if args["df"] is None:
                df = self._read_csv(args["df_path"], usecols=args["smiles_col"])
                num_elements += len(df)
            else:
                num_elements += len(args["df"])
        return num_elements

    ########################### to dict ###############################################
    # Modify these
    def to_dict(self):
        obj_repr = {}
        obj_repr["name"] = self.__class__.__name__
        obj_repr["len"] = len(self)
        obj_repr["train_size"] = len(self.train_indices) if self.train_indices is not None else None
        obj_repr["val_size"] = len(self.val_indices) if self.val_indices is not None else None
        obj_repr["test_size"] = len(self.test_indices) if self.test_indices is not None else None
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
        sample_size: Union[int, float, Type[None]] = None,
        batch_size_train_val: int = 16,
        batch_size_test: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        featurization_backend: str = "loky",
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
            featurization_backend: The backend to use for the molecular featurization.

                - "multiprocessing": Found to cause less memory issues.
                - "loky": joblib's Default. Found to cause memory leaks.
                - "threading": Found to be slow.

            collate_fn: A custom torch collate function. Default is to `goli.data.goli_collate_fn`
            sample_size:

                - `int`: The maximum number of elements to take from the dataset.
                - `float`: Value between 0 and 1 representing the fraction of the dataset to consider
                - `None`: all elements are considered.
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
        dm_args["featurization_backend"] = featurization_backend
        dm_args["persistent_workers"] = persistent_workers
        dm_args["collate_fn"] = collate_fn
        dm_args["weights_col"] = weights_col
        dm_args["weights_type"] = weights_type
        dm_args["sample_size"] = sample_size

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
        if metadata["download_name"].startswith("pcqm4m"):
            df_path = dataset_dir / "raw" / "data.csv.gz"
        else:
            df_path = dataset_dir / "mapping" / "mol.csv.gz"
        logger.info(f"Loading {df_path} in memory.")
        df = pd.read_csv(df_path)

        # Load split from the OGB dataset and save them in a single CSV file
        if metadata["download_name"].startswith("pcqm4m"):
            split_name = metadata["split"]
            split_dict = torch.load(dataset_dir / "split_dict.pt")
            train_split = pd.DataFrame(split_dict["train"])
            val_split = pd.DataFrame(split_dict["valid"])
            if "test" in split_dict.keys():
                test_split = pd.DataFrame(split_dict["test"])
            else:
                test_split = pd.DataFrame(split_dict["test-dev"])

            splits = pd.concat([train_split, val_split, test_split], axis=1)  # type: ignore
            splits.columns = ["train", "val", "test"]

            splits_path = dataset_dir / "split"
            if not splits_path.exists():
                os.makedirs(splits_path)
                splits_path = dataset_dir / f"{split_name}.csv.gz"
            else:
                splits_path = splits_path / f"{split_name}.csv.gz"
            logger.info(f"Saving splits to {splits_path}")
            splits.to_csv(splits_path, index=None)
        else:
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
        if metadata["download_name"].startswith("pcqm4m"):
            idx_col = df.columns[0]
            smiles_col = df.columns[-2]
            label_cols = df.columns[-1:].to_list()
        else:
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

        # Add metadata related to PCQM4M
        ogb_metadata = ogb_metadata.append(pd.DataFrame(PCQM4M_meta, index=["ogbg-lsc-pcqm4m"]))
        ogb_metadata = ogb_metadata.append(pd.DataFrame(PCQM4Mv2_meta, index=["ogbg-lsc-pcqm4mv2"]))

        # Only keep datasets of type 'mol'
        ogb_metadata = ogb_metadata[ogb_metadata["data type"] == "mol"]

        return ogb_metadata
