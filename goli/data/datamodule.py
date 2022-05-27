from typing import Type, List, Dict, Union, Any, Callable, Optional, Tuple, Iterable

import os
from functools import partial
import importlib.resources
import zipfile
from copy import deepcopy
from itertools import cycle
import random

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
from goli.features import mol_to_dglgraph_dict, mol_to_dglgraph_signature, mol_to_dglgraph, dgl_dict_to_graph
from goli.data.collate import goli_collate_fn
from goli.utils.arg_checker import check_arg_iterator

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import Subset

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

class SingleTaskDGLDataset(Dataset):
    """"
    For the MTL pipeline, this replaces (for now) the DGLDataset class, 
    since we do not need to perform featurization straight away. 
    The featurization occurs after gathering all the unique molecules across all datasets.
    """"
    def __init__(
        self,
        labels: Union[torch.Tensor, np.ndarray],
        smiles: Optional[List[str]] = None,
        indices: Optional[List[str]] = None,
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        self.smiles = smiles
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

        datum["labels"] = self.labels[idx]
        return datum

class MultitaskDGLDataset(Dataset):
    """
    This custom dataset combines several datasets into one.
    """
    def __init__(
        self,
        datasets: Dict[str, DGLDataset],
        features = Dict[str, dgl.DGLGraph] # Since we featurize later (happens inside here probably)
    ):
        self.datasets = datasets
        self.features = features
        self.smiles, self.smiles_to_indices = self._get_unique_molecules_and_indices_in_datasets(self.datasets)


    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        datum = {}

        if self.smiles is not None:
            datum["smiles"] = self.smiles[idx]

        # indices
        
        # weights: technically don't need this any longer
        
        # features
        datum["features"] = self.features # featurization happens inside the multitask dataset. Need to modify this
        
        # labels: depend on task
        datum["labels"] = self._get_molecule_labels(self.smiles[idx], self.datasets, self.smiles_to_indices)

        return 0

    # Some helper functions
    def _get_unique_molecules_and_indices_in_datasets(datasets: Dict[str, DGLDataset]):
        """ This function returns all the unique molecules (represented by SMILES) among all datasets """
        # Each dataset is a DGLDataset, so it has lists for each of its properties
        # dataset = {
        #   'smiles' = List[str]
        # }
        # For each dataset, check dataset['smiles']
        #smiles_list = []
        #for dataset in datasets.values():
        #    smiles_list.extend(dataset["smiles"])
        #unique_smiles = list(set(smiles_list)) # Do we want to make this a list?

        # For each dataset, we map a SMILES string to its index, so that later we can easily index the molecule data.
        # This operation is linear in total number of data points across all datasets.
        smiles_to_idx_in_datasets = {}
        for task, dataset in datasets.items():
            #dataset["smiles"] is a list and so it contains the indices implicitly
            smiles_to_idx_in_datasets[task] = {smiles: i for i, smiles in enumerate(dataset["smiles"])}

        # Get the list of unique SMILES strings across all datasets
        smiles_list = []
        for dataset_indices in smiles_to_idx_in_datasets.values():
            smiles_list.extend(dataset_indices.keys())
        unique_smiles = list(set(smiles_list))

        # Now we must return the exact data for each molecule (SMILES).
        unique_smiles_to_idx_in_datasets = {}
        for smiles in unique_smiles:
            task_specific_indices = {}
            for task, dataset_indices in smiles_to_idx_in_datasets.items():
                task_specific_indices[task] = dataset_indices[smiles]
            unique_smiles_to_idx_in_datasets[smiles] = task_specific_indices # task: index

        return unique_smiles, unique_smiles_to_idx_in_datasets

    # More efficient to have a dictionary data structure that has the smiles as keys and value is a dictionary containing the other information.
    # Can probably include fetching other information in this loop
    def _get_molecule_labels(smiles: str, datasets: Dict[str, DGLDataset], smiles_to_idx_dict):
        # labels
        # {
        #   'task1' = list of labels
        #   'task2' = list of labels
        # }
        smiles_indices_across_tasks = smiles_to_idx_dict[smiles] #{task: index}
        labels = {}
        for task, index in smiles_indices_across_tasks.items():
            idx = smiles_indices_across_tasks[task]
            labels[task] = datasets[task].labels[idx]

        return labels


# TODO (Dom):
class MultiTaskDataLoader(DataLoader):
    # Takes a dictionary of datasets.
    # Takes a batch sizes
    # Sample each dataset in proportion of their size
    # If dataset_1 has 10k elements and dataset_2 has 100k elements,
    # then 10 times more elements are sampled from dataset_2.
    # At each iteration, sample a quasi-constant amount from each dataset

    """
    Given Dict['task', DGLDataset], it returns a multi-task dataloader, which uses
    task-proportional sampling to sample different datasets.
    """

    # Moving this below DGLDataset, since Python is executed top-to-bottom.
    def __init__(
        self,
        datasets: Dict[str, DGLDataset],
        batch_size: int,
        shuffle,    # Need to choose whether we shuffle or not given that we are customizing the sampling process.
        drop_last
        #num_workers,
        #collate_fn,
        #pin_memory
    ):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Compute the proportions for sampling from the datasets
        self.dataset_sizes = {}
        for task in datasets:
            self.dataset_sizes[task] = len(datasets[task].features) # Is this actually the size?
        self.sampling_weights = self.task_proportional_sampling(self.dataset_sizes.values())
        #self.data_iters = {k: cycle(v) for k, v in self.datasets.items()} # This makes the dataset iterable, but it already is...
        self.task_names = datasets.keys()

    # Could be changed to sample by temperature or some other way.
    def task_proportional_sampling(self, dataset_sizes):
        total_size = sum(dataset_sizes)
        weights = np.array([(size / total_size) for size in dataset_sizes])
        return weights
    
    # Generate "indices" to iterate over
    def sample_task_data_indices(self):
        random_task_list = []
        for task in self.task_names:
            random_task_list.append(task) * self.dataset_sizes[task]
        random_task_list = random.shuffle(random_task_list)
        sampled_task_data_indices = []
        for t in random_task_list:
            sampled_task_data_indices.append((t, next(self.datasets[t]))) # Get the next datapoint and save it with the task
        return sampled_task_data_indices
            
    def __len__(self):
        return sum(v for k, v in self.dataset_sizes.items()) # Returns the number of data points, after considering all combined datasets

    # Sample each dataset in proportion to the size of the respective dataset
    # We must return the batch in such a way that it preserves the task, so that it can go to the correct output head
    # and the correct metrics can be measured
    def __iter__(self):
        batch = []
        #batch_dict = {} # This stores the datapoints as a list, separated by task
        #for task in self.task_names:
            #batch_dict[task] = []
        indices_sampler = self.sample_task_data_indices()
        for idx in indices_sampler: # Iterate over sampled data points
            batch.append((idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                #for task in self.task_names:
                #    batch_dict[task] = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

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

        # TODO (Dom): Use MultiTaskDataLoader instead.
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

# TODO (Gabriela): Abstract all of prepare_data for a single DGLDataset so that it can be called many times for the MultitaskDGLDataModule
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
        sample_size: Union[int, float, type(None)] = None,
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
        prepare_dict_or_graph: str = "dict",
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

        if prepare_dict_or_graph == "dict":
            self.smiles_transformer = partial(mol_to_dglgraph_dict, **featurization)
        elif prepare_dict_or_graph == "graph":
            self.smiles_transformer = partial(mol_to_dglgraph, **featurization)
        else:
            raise ValueError(
                f"`prepare_dict_or_graph` should be either 'dict' or 'graph', Provided: `{prepare_dict_or_graph}`"
            )

    def prepare_data(self): # Can create train_ds, val_ds and test_ds here instead of setup. Be careful with the featurization (to not compute several times)
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

        # Load the dataframe
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

    def _parse_label_cols(self, label_cols: Union[type(None), str, List[str]], smiles_col: str) -> List[str]:
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
        if "feat" in graph.edata.keys():
            return graph.edata["feat"].shape[1]  # type: ignore
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

        if isinstance(graph, dict):
            graph = dgl_dict_to_graph(**graph, mask_nan=0.0)

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
        np.ndarray, np.ndarray, Union[type(None), np.ndarray], Dict[str, Union[type(None), np.ndarray]]
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


class MTLDGLFromSmilesDataModule(DGLBaseDataModule):
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
        task_specific_df: Dict[str, pd.DataFrame] = None,         # task dict
        task_specific_df_path: Optional[Dict[str, Union[str, os.PathLike]]] = None,  # task dict
        task_specific_smiles_col: Dict[str, str] = None,         # task dict
        task_specific_label_col: Dict[str, List[str]] = None,   # task dict
        task_specific_weights_col: Dict[str, str] = None,        # task dict
        task_specific_weights_type: Dict[str, str] = None,       # task dict
        task_specific_idx_col: Dict[str, str] = None,            # task dict
        task_specific_sample_size: Dict[str, Union[int, float, None]] = None,      # task dict
        task_specific_split_val: Dict[str, float] = None,         # task dict
        task_specific_split_test: Dict[str, float]= None,        # task dict
        task_specific_split_seed: Dict[str, int] = None,         # task dict
        task_specific_splits_path: Optional[Dict[str, Union[str, os.PathLike]]] = None,  # task dict
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
        collate_fn: Optional[Callable] = None,      # need new one
        prepare_dict_or_graph: str = "dict",
        dataset_class: type = MultitaskDGLDataset,
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
                  and requires more RAM, but faster during training.
                - "dict": Process molecules as a Dict. It's faster and requires less RAM during
                  pre-processing, but slower during training since DGLGraphs will be created
                  during data-loading.
            dataset_class: The class used to create the dataset from which to sample.
        """
        # We create a DGLBaseDataModule type here.
        super().__init__(
            batch_size_train_val=batch_size_train_val,
            batch_size_test=batch_size_test,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn, 
        )

        # Just store all the input arguments for creating the basic object. 
        # All these arguments are relevant for creating the single-task dataset.
        self.df = task_specific_df
        self.df_path = task_specific_df_path
        self.smiles_col = task_specific_smiles_col
        self.label_cols = {task: self._parse_label_cols(task_specific_label_col[task], task_specific_smiles_col[task]) for task in self.df.keys()}
        self.idx_col = task_specific_idx_col
        self.sample_size = task_specific_sample_size

        self.weights_col = task_specific_weights_col
        self.weights_type = task_specific_weights_type
        if self.weights_col is not None:
            assert self.weights_type is None

        # Keep the splits in mind for MTL!
        self.split_val = task_specific_split_val
        self.split_test = task_specific_split_test
        self.split_seed = task_specific_split_seed
        self.splits_path = task_specific_splits_path

        # Here we take care of caching and featurization arguments
        self.cache_data_path = str(cache_data_path) if cache_data_path is not None else None
        self.featurization = featurization if featurization is not None else {}


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

        if prepare_dict_or_graph == "dict":
            self.smiles_transformer = partial(mol_to_dglgraph_dict, **featurization)
        elif prepare_dict_or_graph == "graph":
            self.smiles_transformer = partial(mol_to_dglgraph, **featurization)
        else:
            raise ValueError(
                f"`prepare_dict_or_graph` should be either 'dict' or 'graph', Provided: `{prepare_dict_or_graph}`"
            )

    def prepare_single_task_data(
        self, 
        df: pd.DataFrame = None,
        df_path: Optional[Union[str, os.PathLike]] = None,
        smiles_col: str = None,
        label_cols: List[str] = None,
        weights_col: str = None,
        weights_type: str = None,
        idx_col: str = None
        ):
        """"
        This functions uses the logic for a single task from the original data module.
        The input parameters relate to everything necessary for creating a single task dataset.
        This function generates a SingleTaskDGLDataset for the given dataset.
        Notably, we skip computing features here. Caching also occurs for the multi-task dataset.

        - Load the dataframe
        - Extract smiles and labels from the dataframe.
        - Compute or set split indices.

        Called only from a single process in distributed settings. Steps:

        - If cache is set and exists, reload from cache.
        - Load the dataframe if it's a path.
        - Extract smiles and labels from the dataframe.
        - Compute the features.
        - Compute or set split indices.
        - Make the list of dict dataset.                
        """

        # Load the dataframe
        if df is None:
            # Only load the useful columns, as some datasets can be very large
            # when loading all columns
            label_columns = check_arg_iterator(label_cols, enforce_type=list)
            usecols = (
                check_arg_iterator(smiles_col, enforce_type=list)
                + label_columns
                + check_arg_iterator(idx_col, enforce_type=list) # No need to add the weights
            )
            label_dtype = {col: np.float16 for col in label_columns}

            data_frame = self._read_csv(df_path, usecols=usecols, dtype=label_dtype)
        else:
            data_frame = df
        
        data_frame = self._sub_sample_df(data_frame)

        logger.info(f"Prepare single-task dataset with {len(data_frame)} data points.")

        # Extract smiles and labels
        smiles, labels, sample_idx, extras = self._extract_smiles_labels(
            df,
            smiles_col=smiles_col,
            label_cols=label_cols,
            idx_col=idx_col,
            weights_col=weights_col,
            weights_type=weights_type,
        )

        return data_frame, smiles, labels, sample_idx, extras

    def prepare_data(self):
        """Called only from a single process in distributed settings. Steps:
        
        - If cache is set and exists, reload from cache.
        - Generate single task datasets.
        - Compute the features.
        - Combine single task datasets into multi-task dataset.
        """

        # Load from cache. Need to make some changes here.


        # Load the single-task dataframes
        single_task_data_dict = {}
        single_task_smiles_labels_extras = {}
        for task in self.df.keys():
            smiles_labels_extras = {}
            single_task_data_dict[task], smiles_labels_extras['smiles'], smiles_labels_extras['labels'], smiles_labels_extras['sample_idx'], smiles_labels_extras['extras'] = self.prepare_single_task_data(
                df=self.df[task],
                df_path=self.df_path[task],
                smiles_col=self.smiles_col[task],
                labels_cols=self.label_cols[task],
                weights_col=self.weights_col[task],
                weights_type=self.weights_type[task],
                idx_col=self.idx_col[task]
            )
            single_task_smiles_labels_extras[task] = smiles_labels_extras

        # Get the unique smiles in order to featurize
        unique_smiles, unique_smiles_to_idx_in_datasets = self._get_unique_molecules_and_indices_in_datasets(single_task_data_dict)

        # Featurize the molecules
        features, idx_none = self._featurize_molecules(unique_smiles)

        # Filter the molecules, labels, etc. for the molecules that failed featurization
        for task in single_task_data_dict:
            single_task_data_dict[task], single_task_smiles_labels_extras[task]['features'], single_task_smiles_labels_extras[task]['smiles'], single_task_smiles_labels_extras[task]['labels'], single_task_smiles_labels_extras[task]['sample_idx'], single_task_smiles_labels_extras[task]['extras'] = self._filter_none_molecules(
                idx_none, single_task_data_dict[task], features, single_task_smiles_labels_extras[task]['smiles'], single_task_smiles_labels_extras[task]['labels'], single_task_smiles_labels_extras[task]['sample_idx'], single_task_smiles_labels_extras[task]['extras']
            )

        # Get splits indices
        self.train_indices = {}
        self.val_indices = {}
        self.test_indices = {}
        for task in single_task_data_dict:
            self.train_indices[task], self.val_indices[task], self.test_indices[task] = self._get_split_indices(
                len(single_task_data_dict[task]),
                split_val=self.split_val[task],
                split_test=self.split_test[task],
                split_seed=self.split_seed[task],
                splits_path=self.splits_path[task],
                sample_idx=single_task_smiles_labels_extras['sample_idx']
            )

        # Split the dict of task-specific datasets into three: one for training, one for val, one for test
        train_single_task_dataset_dict = {}
        val_single_task_dataset_dict = {}
        test_single_task_dataset_dict = {}
        for task in single_task_data_dict:
            train_single_task_dataset_dict[task] = Subset(single_task_data_dict[task], self.train_indices[task])
            val_single_task_dataset_dict[task] = Subset(single_task_data_dict[task], self.val_indices[task])
            test_single_task_dataset_dict[task] = Subset(single_task_data_dict[task], self.test_indices[task])


        # Create 3 multitask datasets: one for train, one for val, one for test
        self.train_ds = MultitaskDGLDataset(train_single_task_dataset_dict)
        self.val_ds = MultitaskDGLDataset(val_single_task_dataset_dict)
        self.test_ds = MultitaskDGLDataset(test_single_task_dataset_dict)


    def _get_unique_molecules_and_indices_in_datasets(datasets: Dict[str, pd.DataFrame]):
        """ This function returns all the unique molecules (represented by SMILES) among all datasets """

        smiles_to_idx_in_datasets = {}
        for task, dataset in datasets.items():
            smiles_to_idx_in_datasets[task] = {smiles: i for i, smiles in enumerate(dataset["smiles"])}

        # Get the list of unique SMILES strings across all datasets
        smiles_list = []
        for dataset_indices in smiles_to_idx_in_datasets.values():
            smiles_list.extend(dataset_indices.keys())
        unique_smiles = list(set(smiles_list))

        # Now we must return the exact data for each molecule (SMILES).
        unique_smiles_to_idx_in_datasets = {}
        for smiles in unique_smiles:
            task_specific_indices = {}
            for task, dataset_indices in smiles_to_idx_in_datasets.items():
                task_specific_indices[task] = dataset_indices[smiles]
            unique_smiles_to_idx_in_datasets[smiles] = task_specific_indices # task: index

        return unique_smiles, unique_smiles_to_idx_in_datasets


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

    def _parse_label_cols(
        self, 
        df: pd.DataFrame, 
        str: Optional[Union[str, os.PathLike]], 
        label_cols: Union[type(None), str, List[str]], 
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

        if isinstance(graph, dict):
            graph = dgl_dict_to_graph(**graph, mask_nan=0.0)

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
        np.ndarray, np.ndarray, Union[type(None), np.ndarray], Dict[str, Union[type(None), np.ndarray]]
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






# TODO (Dom):
class MultiTaskDGLFromSmilesDataModule(pl.LightningDataModule):
    pass
    # Take a dict of parameters for DGLFromSmilesDataModule
    # Initialize many DGLFromSmilesDataModule.
    # When `setup` and `prepare_data` are called, call it for all DGLFromSmilesDataModule
    # ALSO!! check other functions needed by pl.Trainer to make it work with the Predictor. Maybe just the Dataloader???

    def __init__(
        self, 
        task_dglfromsmilesdatamodule_kwargs: Dict[str, Any]
        ):
        data_modules = {}
        for task in task_dglfromsmilesdatamodule_kwargs:
            data_modules[task] = DGLFromSmilesDataModule(task_dglfromsmilesdatamodule_kwargs[task])
        self.data_modules[task] = data_modules

    def prepare_data(self):
        for task in self.data_modules:
            self.data_modules[task].prepare_data()
    
    def setup(self):
        for task in self.data_modules:
            self.data_modules[task].setup()

    def train_dataloader(self):
        return MultiTaskDataLoader(self.data_modules)
        

class DGLOGBDataModule(DGLFromSmilesDataModule):
    """Load an OGB GraphProp dataset."""

    def __init__(
        self,
        dataset_name: str,
        cache_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        weights_col: str = None,
        weights_type: str = None,
        sample_size: Union[int, float, type(None)] = None,
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
