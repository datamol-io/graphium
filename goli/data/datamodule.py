from typing import Type, List, Dict, Union, Any, Callable, Optional, Tuple, Iterable

import os
from functools import partial
import importlib.resources
import zipfile
from copy import deepcopy
from multiprocessing import Manager
import time

from loguru import logger
import fsspec
import omegaconf

import pandas as pd
import numpy as np
import datamol as dm

from sklearn.model_selection import train_test_split

import dgl
from torch_geometric.data import Data, Batch
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import RunningStage

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import Subset
from functools import lru_cache

from goli.utils import fs
from goli.features import (
    mol_to_graph_dict,
    mol_to_dglgraph,
    GraphDict,
    mol_to_pyggraph,
)
from goli.data.collate import goli_collate_fn
from goli.utils.arg_checker import check_arg_iterator
from goli.utils.hashing import get_md5_hash


PCQM4M_meta = {
    "num tasks": 1,
    "eval metric": "mae",
    "download_name": "pcqm4m_kddcup2021",
    "url": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m_kddcup2021.zip",  # TODO: Allow PyG
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
        "url": "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip",  # TODO: Allow PyG
        "version": 2,
    }
)


def smiles_to_unique_mol_id(smiles: str) -> Optional[str]:
    """
    Convert a smiles to a unique MD5 Hash ID. Returns None if featurization fails.
    Parameters:
        smiles: A smiles string to be converted to a unique ID
    Returns:
        mol_id: a string unique ID
    """
    try:
        mol = dm.to_mol(mol=smiles)
        mol_id = dm.unique_id(mol)
    except:
        mol_id = ""
    if mol_id is None:
        mol_id = ""
    return mol_id


def did_featurization_fail(features: Any) -> bool:
    """
    Check if a featurization failed.
    """
    return (features is None) or isinstance(features, str)


class BatchingSmilesTransform:
    """
    Class to transform a list of smiles using a transform function
    """

    def __init__(self, transform: Callable):
        """
        Parameters:
            transform: Callable function to transform a single smiles
        """
        self.transform = transform

    def __call__(self, smiles_list: Iterable[str]) -> Any:
        """
        Function to transform a list of smiles
        """
        mol_id_list = []
        for smiles in smiles_list:
            mol_id_list.append(self.transform(smiles))
        return mol_id_list

    @staticmethod
    def parse_batch_size(numel: int, desired_batch_size: int, n_jobs: int) -> int:
        """
        Function to parse the batch size.
        The batch size is limited by the number of elements divided by the number of jobs.
        """
        assert ((n_jobs >= 0) or (n_jobs == -1)) and isinstance(
            n_jobs, int
        ), f"n_jobs must be a positive integer or -1, got {n_jobs}"
        assert (
            isinstance(desired_batch_size, int) and desired_batch_size >= 0
        ), f"desired_batch_size must be a positive integer, got {desired_batch_size}"

        if n_jobs == -1:
            n_jobs = os.cpu_count()
        if n_jobs == 0:
            batch_size = numel
        else:
            batch_size = min(desired_batch_size, numel // n_jobs)
        batch_size = max(1, batch_size)
        return batch_size


def smiles_to_unique_mol_ids(
    smiles: Iterable[str],
    n_jobs=-1,
    featurization_batch_size=1000,
    backend="loky",
    progress=True,
    progress_desc="mols to ids",
) -> List[Optional[str]]:
    """
    This function takes a list of smiles and finds the corresponding datamol unique_id
    in an element-wise fashion, returning the corresponding unique_ids.

    The ID is an MD5 hash of the non-standard InChiKey provided
    by `dm.to_inchikey_non_standard()`. It guarantees uniqueness for
    different tautomeric forms of the same molecule.

    Parameters:
        smiles: a list of smiles to be converted to mol ids
        n_jobs: number of jobs to run in parallel
        backend: Parallelization backend
        progress: Whether to display the progress bar

    Returns:
        ids: A list of MD5 hash ids
    """

    batch_size = BatchingSmilesTransform.parse_batch_size(
        numel=len(smiles), desired_batch_size=featurization_batch_size, n_jobs=n_jobs
    )

    unique_mol_ids = dm.parallelized_with_batches(
        BatchingSmilesTransform(smiles_to_unique_mol_id),
        smiles,
        batch_size=batch_size,
        progress=progress,
        n_jobs=n_jobs,
        backend=backend,
        tqdm_kwargs={"desc": f"{progress_desc}, batch={batch_size}"},
    )

    return unique_mol_ids


class SingleTaskDataset(Dataset):
    def __init__(
        self,
        labels: Union[torch.Tensor, np.ndarray],
        features: Optional[List[Union[dgl.DGLGraph, GraphDict]]] = None,
        smiles: Optional[List[str]] = None,
        indices: Optional[List[str]] = None,
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        unique_ids: Optional[List[str]] = None,
    ):
        r"""
        dataset for a single task
        Parameters:
            labels: A list of labels for the given task (one per graph)
            features: A list of graphs
            smiles: A list of smiles
            indices: A list of indices
            weights: A list of weights
            unique_ids: A list of unique ids
        """
        self.labels = labels
        if smiles is not None:
            manager = Manager()  # Avoid memory leaks with `num_workers > 0` by using the Manager
            self.smiles = manager.list(smiles)
        else:
            self.smiles = None
        self.features = features
        self.indices = indices
        if self.indices is not None:
            self.indices = np.array(
                self.indices
            )  # Avoid memory leaks with `num_workers > 0` by using numpy array
        self.weights = weights
        self.unique_ids = unique_ids

    def __len__(self):
        r"""
        return the size of the dataset
        Returns:
            size: the size of the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        get the data at the given index
        Parameters:
            idx: the index to get the data at
        Returns:
            datum: a dictionary containing the data at the given index, with keys "features", "labels", "smiles", "indices", "weights", "unique_ids"
        """
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

        if self.unique_ids is not None:
            datum["unique_ids"] = self.unique_ids[idx]

        return datum

    def __getstate__(self):
        """Serialize the class for pickling."""
        state = {}
        state["labels"] = self.labels
        state["smiles"] = list(self.smiles) if self.smiles is not None else None
        state["features"] = self.features
        state["indices"] = self.indices
        state["weights"] = self.weights
        state["unique_ids"] = self.unique_ids
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        if state["smiles"] is not None:
            manager = Manager()
            state["smiles"] = manager.list(state["smiles"])

        self.__dict__.update(state)


class MultitaskDataset(Dataset):
    def __init__(
        self,
        datasets: Dict[str, SingleTaskDataset],
        n_jobs=-1,
        backend: str = "loky",
        featurization_batch_size=1000,
        progress: bool = True,
        about: str = "",
        generated: bool = False,
    ):
        r"""
        This class holds the information for the multitask dataset.
        Several single-task datasets can be merged to create a multi-task dataset. After merging the dictionary of single-task datasets.
        we will have a multitask dataset of the following form:
        - self.mol_ids will be a list to contain the unique molecular IDs to identify the molecules
        - self.smiles will be a list to contain the corresponding smiles for that molecular ID across all single-task datasets
        - self.labels will be a list of dictionaries where the key is the task name and the value is the label(s) for that task.
            At this point, any particular molecule will only have entries for tasks for which it has a label. Later, in the collate
            function, we fill up the missing task labels with NaNs.
        - self.features will be a list of featurized graphs corresponding to that particular unique molecule.
            However, for testing purposes we may not require features so that we can make sure that this merge function works.

        Parameters:
            datasets: A dictionary of single-task datasets
            n_jobs: Number of jobs to run in parallel
            backend: Parallelization backend
        progress: Whether to display the progress bar
            about: A description of the dataset
        generated: bool = False,
        """
        super().__init__()
        # self.datasets = datasets
        self.generated = generated
        self.n_jobs = n_jobs
        self.backend = backend
        self.featurization_batch_size = featurization_batch_size
        self.progress = progress
        self.about = about

        task = next(iter(datasets))
        if "features" in datasets[task][0]:
            self.mol_ids, self.smiles, self.labels, self.features = self.merge(datasets)
        else:
            self.mol_ids, self.smiles, self.labels = self.merge(datasets)

        self.labels = np.array(self.labels)
        self.labels_size = self.set_label_size_dict(datasets)

    def __len__(self):
        r"""
        Returns the number of molecules
        """
        return len(self.labels)

    @property
    def num_graphs_total(self):
        r"""
        number of graphs (molecules) in the dataset
        """
        return len(self)

    @property
    def num_nodes_total(self):
        """Total number of nodes for all graphs"""
        return sum([get_num_nodes(data) for data in self.features])

    @property
    def max_num_nodes_per_graph(self):
        """Maximum number of nodes per graph"""
        return max([get_num_nodes(data) for data in self.features])

    @property
    def std_num_nodes_per_graph(self):
        """Standard deviation of number of nodes per graph"""
        return np.std([get_num_nodes(data) for data in self.features])

    @property
    def min_num_nodes_per_graph(self):
        """Minimum number of nodes per graph"""
        return min([get_num_nodes(data) for data in self.features])

    @property
    def mean_num_nodes_per_graph(self):
        """Average number of nodes per graph"""
        return self.num_nodes_total / self.num_graphs_total

    @property
    def num_edges_total(self):
        """Total number of edges for all graphs"""
        return sum([get_num_edges(data) for data in self.features])

    @property
    def max_num_edges_per_graph(self):
        """Maximum number of edges per graph"""
        return max([get_num_edges(data) for data in self.features])

    @property
    def min_num_edges_per_graph(self):
        """Minimum number of edges per graph"""
        return min([get_num_edges(data) for data in self.features])

    @property
    def std_num_edges_per_graph(self):
        """Standard deviation of number of nodes per graph"""
        return np.std([get_num_edges(data) for data in self.features])

    @property
    def mean_num_edges_per_graph(self):
        """Average number of edges per graph"""
        return self.num_edges_total / self.num_graphs_total

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        r"""
        get the data for at the specified index
        Parameters:
            idx: The index of the data to retrieve
        Returns:
            A dictionary containing the data for the specified index with keys "mol_ids", "smiles", "labels", and "features"
        """
        datum = {}

        # Remove mol_ids and smiles for now, to reduce memory consumption b
        # if self.mol_ids is not None:
        #     datum["mol_ids"] = self.mol_ids[idx]

        # if self.smiles is not None:
        #     datum["smiles"] = self.smiles[idx]

        if self.labels is not None:
            datum["labels"] = self.labels[idx]

        if self.features is not None:
            datum["features"] = self.features[idx]

        return datum

    def merge(self, datasets: Dict[str, Any]) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Any]]:
        r"""This function merges several single task datasets into a multitask dataset.

        The idea: for each of the smiles, labels, features and tasks, we create a corresponding list that concatenates these items across all tasks.
        In particular, for any index, the elements in the smiles, labels, features and task lists at that index will correspond to each other (i.e. match up).
        Over this list of all smiles (which we created by concatenating the smiles across all tasks), we compute their molecular ID using functions from Datamol.
        Once again, we will have a list of molecular IDs which is the same size as the list of smiles, labels, features and tasks.
        We then use numpy's `unique` function to find the exact list of unique molecular IDs as these will identify the molecules in our dataset. We also get the
        inverse from numpy's `unique`, which will allow us to index in addition to the list of all molecular IDs, the list of all smiles, labels, features and tasks.
        Finally, we use this inverse to construct the list of list of smiles, list of label dictionaries (indexed by task) and the list of features such that
        the indices match up. This is what is needed for the `get_item` function to work.

        Parameters:
            datasets: A dictionary of single-task datasets
        Returns:
            A tuple of (list of molecular IDs, list of smiles, list of label dictionaries, list of features)
        """
        all_smiles = []
        all_features = []
        all_labels = []
        all_mol_ids = []
        all_tasks = []

        for task, ds in datasets.items():
            # Get data from single task dataset
            ds_smiles = [ds[i]["smiles"] for i in range(len(ds))]
            ds_labels = [ds[i]["labels"] for i in range(len(ds))]
            if "unique_ids" in ds[0].keys():
                ds_mol_ids = [ds[i]["unique_ids"] for i in range(len(ds))]
            else:
                ds_mol_ids = smiles_to_unique_mol_ids(
                    ds_smiles,
                    n_jobs=self.n_jobs,
                    featurization_batch_size=self.featurization_batch_size,
                    backend=self.backend,
                    progress=self.progress,
                    progress_desc=f"{task}: mol to ids",
                )
            if "features" in ds[0]:
                ds_features = [ds[i]["features"] for i in range(len(ds))]
            else:
                ds_features = None

            all_smiles.extend(ds_smiles)
            all_labels.extend(ds_labels)
            all_mol_ids.extend(ds_mol_ids)
            if ds_features is not None:
                all_features.extend(ds_features)

            task_list = [task] * ds.__len__()
            all_tasks.extend(task_list)

        if self.generated is False:
            # Get all unique mol ids.
            unique_mol_ids, inv = np.unique(all_mol_ids, return_inverse=True)
            mol_ids = unique_mol_ids
        else:
            # The generated data is a single molecule duplicated
            mol_ids = np.array(all_mol_ids)
            inv = [_ for _ in range(len(mol_ids))]

        # Store the smiles.
        smiles = [[] for _ in range(len(mol_ids))]
        for all_idx, unique_idx in enumerate(inv):
            smiles[unique_idx].append(all_smiles[all_idx])

        # Store the labels.
        labels = [{} for _ in range(len(mol_ids))]
        for all_idx, unique_idx in enumerate(inv):
            task = all_tasks[all_idx]
            label = all_labels[all_idx]
            labels[unique_idx][task] = label

        # Store the features
        if len(all_features) > 0:
            features = [-1 for i in range(len(mol_ids))]
            for all_idx, unique_idx in enumerate(inv):
                features[unique_idx] = all_features[all_idx]
            return mol_ids, smiles, labels, features
        else:
            return mol_ids, smiles, labels

    def set_label_size_dict(self, datasets: Dict[str, SingleTaskDataset]):
        r"""
        This gives the number of labels to predict for a given task.
        """
        task_labels_size = {}
        for task, ds in datasets.items():
            label = ds[0][
                "labels"
            ]  # Assume for a fixed task, the label dimension is the same across data points, so we can choose the first data point for simplicity.
            torch_label = torch.as_tensor(label)
            # torch_label = label
            task_labels_size[task] = torch_label.size()
        return task_labels_size

    def __repr__(self) -> str:
        """
        summarizes the dataset in a string
        Returns:
            A string representation of the dataset.
        """
        out_str = (
            f"-------------------\n{self.__class__.__name__}\n"
            + f"\tabout = {self.about}\n"
            + f"\tnum_graphs_total = {self.num_graphs_total}\n"
            + f"\tnum_nodes_total = {self.num_nodes_total}\n"
            + f"\tmax_num_nodes_per_graph = {self.max_num_nodes_per_graph}\n"
            + f"\tmin_num_nodes_per_graph = {self.min_num_nodes_per_graph}\n"
            + f"\tstd_num_nodes_per_graph = {self.std_num_nodes_per_graph}\n"
            + f"\tmean_num_nodes_per_graph = {self.mean_num_nodes_per_graph}\n"
            + f"\tnum_edges_total = {self.num_edges_total}\n"
            + f"\tmax_num_edges_per_graph = {self.max_num_edges_per_graph}\n"
            + f"\tmin_num_edges_per_graph = {self.min_num_edges_per_graph}\n"
            + f"\tstd_num_edges_per_graph = {self.std_num_edges_per_graph}\n"
            + f"\tmean_num_edges_per_graph = {self.mean_num_edges_per_graph}\n"
            + f"-------------------\n"
        )
        return out_str


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        collate_fn: Optional[Callable] = None,
    ):
        """
        base dataset module for all datasets (to be inherented)

        Parameters:
            batch_size_training: batch size for training
            batch_size_inference: batch size for inference
            num_workers: number of workers for data loading
            pin_memory: whether to pin memory
            persistent_workers: whether to use persistent workers
            collate_fn: collate function for batching
        """
        super().__init__()

        self.batch_size_training = batch_size_training
        self.batch_size_inference = batch_size_inference

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self.collate_fn = self.get_collate_fn(collate_fn)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._predict_ds = None

        self._data_is_prepared = False

    def prepare_data(self):
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def train_dataloader(self, **kwargs):
        """
        return the training dataloader
        """
        return self.get_dataloader(
            dataset=self.train_ds,  # type: ignore
            shuffle=True,
            stage=RunningStage.TRAINING,
            **kwargs,
        )

    def val_dataloader(self, **kwargs):
        r"""
        return the validation dataloader
        """
        return self.get_dataloader(
            dataset=self.val_ds,  # type: ignore
            shuffle=False,
            stage=RunningStage.VALIDATING,
            **kwargs,
        )

    def test_dataloader(self, **kwargs):
        r"""
        return the test dataloader
        """
        return self.get_dataloader(
            dataset=self.test_ds,  # type: ignore
            shuffle=False,
            stage=RunningStage.TESTING,
            **kwargs,
        )

    def predict_dataloader(self, **kwargs):
        """
        return the dataloader for prediction
        """
        return self.get_dataloader(
            dataset=self.predict_ds,  # type: ignore
            shuffle=False,
            stage=RunningStage.PREDICTING,
            **kwargs,
        )

    @staticmethod
    def get_collate_fn(collate_fn):
        if collate_fn is None:
            # Some values become `inf` when changing data type. `mask_nan` deals with that
            collate_fn = partial(goli_collate_fn, mask_nan=0)
            collate_fn.__name__ = goli_collate_fn.__name__

        return collate_fn

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

    @property
    def get_num_workers(self):
        """
        get the number of workers to use
        """
        if self.num_workers == -1:
            num_workers = os.cpu_count()
            num_workers = num_workers if num_workers is not None else 0
        else:
            num_workers = self.num_workers
        return num_workers

    @predict_ds.setter
    def predict_ds(self, value):
        """Set the dataset for the prediction"""
        self._predict_ds = value

    def get_first_graph(self):
        raise NotImplementedError()

    # Private methods

    @staticmethod
    def _read_csv(
        path: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        private method for reading a csv file
        Parameters:
            path: path to the csv file
            kwargs: keyword arguments for pd.read_csv
        Returns:
            pd.DataFrame: the panda dataframe storing molecules
        """

        if str(path).endswith((".csv", ".csv.gz", ".csv.zip", ".csv.bz2")):
            sep = ","
        elif str(path).endswith((".tsv", ".tsv.gz", ".tsv.zip", ".tsv.bz2")):
            sep = "\t"
        else:
            raise ValueError(f"unsupported file `{path}`")
        kwargs.setdefault("sep", sep)
        df = pd.read_csv(path, **kwargs)
        return df

    @staticmethod
    def _read_parquet(path: str, **kwargs) -> pd.DataFrame:
        """
        read the parquet file int a pandas dataframe
        Parameters:
            path: path to the parquet file
            kwargs: keyword arguments for pd.read_parquet
        Returns:
            pd.DataFrame: the panda dataframe storing molecules
        """
        df = pd.read_parquet(path)
        return df

    @staticmethod
    def _read_table(self, path: str, **kwargs) -> pd.DataFrame:
        """
        a general read file function which determines if which function to use, either _read_csv or _read_parquet
        Parameters:
            path: path to the file to read
            kwargs: keyword arguments for pd.read_csv or pd.read_parquet
        Returns:
            pd.DataFrame: the panda dataframe storing molecules
        """
        if str(path).endswith((".parquet")):
            return self._read_parquet(path)
        else:
            return self._read_csv(path)

    def get_dataloader_kwargs(self, stage: RunningStage, shuffle: bool, **kwargs) -> Dict[str, Any]:
        """
        Get the options for the dataloader depending on the current stage.

        Parameters:
            stage: Whether in Training, Validating, Testing, Sanity-checking, Predicting, or Tuning phase.
            shuffle: set to ``True`` to have the data reshuffled at every epoch.

        Returns:
            Arguments to pass to the `DataLoader` during initialization
        """
        loader_kwargs = {}

        # Get batch size and IPU options for training set
        # if stage in [RunningStage.TRAINING, RunningStage.TUNING]:
        if stage in [RunningStage.TRAINING]:
            loader_kwargs["batch_size"] = self.batch_size_training

        # Get batch size and IPU options for validation / testing sets
        elif stage in [RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]:
            loader_kwargs["batch_size"] = self.batch_size_inference
        else:
            raise ValueError(f"Wrong value for `stage`. Provided `{stage}`")

        # Set default parameters
        loader_kwargs["shuffle"] = shuffle
        loader_kwargs["collate_fn"] = self.collate_fn
        loader_kwargs["num_workers"] = self.get_num_workers
        loader_kwargs["pin_memory"] = self.pin_memory
        loader_kwargs["persistent_workers"] = self.persistent_workers

        # Update from provided parameters
        loader_kwargs.update(**kwargs)

        return loader_kwargs

    def get_dataloader(self, dataset: Dataset, shuffle: bool, stage: RunningStage) -> DataLoader:
        """
        Get the dataloader for a given dataset

        Parameters:
            dataset: The dataset from which to load the data
            shuffle: set to ``True`` to have the data reshuffled at every epoch.
            stage: Whether in Training, Validating, Testing, Sanity-checking, Predicting, or Tuning phase.

        Returns:
            The dataloader to sample from
        """
        kwargs = self.get_dataloader_kwargs(stage=stage, shuffle=shuffle)
        return self._dataloader(dataset=dataset, shuffle=shuffle, stage=stage, **kwargs)

    def _dataloader(self, dataset: Dataset, **kwargs) -> DataLoader:
        r"""
        Get a dataloader for a given dataset
        Parameters:
            dataset: The dataset from which to load the data
            kwargs: keyword arguments for DataLoader
        Returns:
            The dataloader to sample from
        """

        loader = DataLoader(
            dataset=dataset,
            **kwargs,
        )

        return loader

    def get_max_num_nodes_datamodule(self, stages: Optional[List[str]] = None) -> int:
        """
        Get the maximum number of nodes across all datasets from the datamodule

        Parameters:
            datamodule: The datamodule from which to extract the maximum number of nodes
            stages: The stages from which to extract the max num nodes.
                Possible values are ["train", "val", "test", "predict"].
                If None, all stages are considered.

        Returns:
            max_num_nodes: The maximum number of nodes across all datasets from the datamodule
        """

        allowed_stages = ["train", "val", "test", "predict"]
        if stages is None:
            stages = allowed_stages
        for stage in stages:
            assert stage in allowed_stages, f"stage value `{stage}` not allowed."

        max_num_nodes = 0
        # Max number of nodes in the training dataset
        if (self.train_ds is not None) and ("train" in stages):
            max_num_nodes = max(max_num_nodes, self.train_ds.max_num_nodes_per_graph)

        # Max number of nodes in the validation dataset
        if (self.val_ds is not None) and ("val" in stages):
            max_num_nodes = max(max_num_nodes, self.val_ds.max_num_nodes_per_graph)

        # Max number of nodes in the test dataset
        if (self.test_ds is not None) and ("test" in stages):
            max_num_nodes = max(max_num_nodes, self.test_ds.max_num_nodes_per_graph)

        # Max number of nodes in the predict dataset
        if (self.predict_ds is not None) and ("predict" in stages):
            max_num_nodes = max(max_num_nodes, self.predict_ds.max_num_nodes_per_graph)

        return max_num_nodes

    def get_max_num_edges_datamodule(self, stages: Optional[List[str]] = None) -> int:
        """
        Get the maximum number of edges across all datasets from the datamodule

        Parameters:
            datamodule: The datamodule from which to extract the maximum number of nodes
            stages: The stages from which to extract the max num nodes.
                Possible values are ["train", "val", "test", "predict"].
                If None, all stages are considered.

        Returns:
            max_num_edges: The maximum number of edges across all datasets from the datamodule
        """

        allowed_stages = ["train", "val", "test", "predict"]
        if stages is None:
            stages = allowed_stages
        for stage in stages:
            assert stage in allowed_stages, f"stage value `{stage}` not allowed."

        max_num_edges = 0
        # Max number of nodes/edges in the training dataset
        if (self.train_ds is not None) and ("train" in stages):
            max_num_edges = max(max_num_edges, self.train_ds.max_num_edges_per_graph)

        # Max number of nodes/edges in the validation dataset
        if (self.val_ds is not None) and ("val" in stages):
            max_num_edges = max(max_num_edges, self.val_ds.max_num_edges_per_graph)

        # Max number of nodes/edges in the test dataset
        if (self.test_ds is not None) and ("test" in stages):
            max_num_edges = max(max_num_edges, self.test_ds.max_num_edges_per_graph)

        # Max number of nodes/edges in the predict dataset
        if (self.predict_ds is not None) and ("predict" in stages):
            max_num_edges = max(max_num_edges, self.predict_ds.max_num_edges_per_graph)

        return max_num_edges


class DatasetProcessingParams:
    def __init__(
        self,
        df: pd.DataFrame = None,
        df_path: Optional[Union[str, os.PathLike]] = None,
        smiles_col: str = None,
        label_cols: List[str] = None,
        weights_col: str = None,  # Not needed
        weights_type: str = None,  # Not needed
        idx_col: str = None,
        sample_size: Union[int, float, Type[None]] = None,
        split_val: float = 0.2,
        split_test: float = 0.2,
        split_seed: int = None,
        splits_path: Optional[Union[str, os.PathLike]] = None,
        generated_data: bool = False,
    ):
        """
        object to store the parameters for the dataset processing
        Parameters:
            df: The dataframe containing the data
            df_path: The path to the dataframe containing the data
            smiles_col: The column name of the smiles
            label_cols: The column names of the labels
            weights_col: The column name of the weights
            weights_type: The type of weights
            idx_col: The column name of the indices
            sample_size: The size of the sample
            split_val: The fraction of the data to use for validation
            split_test: The fraction of the data to use for testing
            split_seed: The seed to use for the split
            splits_path: The path to the splits
        """
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
        self.generated_data = generated_data


class IPUDataModuleModifier:
    def __init__(
        self,
        ipu_inference_opts: Optional["poptorch.Options"] = None,
        ipu_training_opts: Optional["poptorch.Options"] = None,
        ipu_dataloader_training_opts: Optional["IPUDataloaderOptions"] = None,
        ipu_dataloader_inference_opts: Optional["IPUDataloaderOptions"] = None,
        *args,
        **kwargs,
    ) -> None:
        r"""
        wrapper functions from the a `DataModule` to support IPU and IPU options To be used in dual inheritance, for example:
        ```
        IPUDataModule(BaseDataModule, IPUDataModuleModifier):
            def __init__(self, **kwargs):
                BaseDataModule.__init__(self, **kwargs)
                IPUDataModuleModifier.__init__(self, **kwargs)
        ```

        Parameters:
            ipu_inference_opts: Options for the IPU in inference mode. Ignore if not using IPUs
            ipu_training_opts: Options for the IPU in training mode. Ignore if not using IPUs
            ipu_dataloader_kwargs_train_val: Options for the dataloader for the IPU. Ignore if not using IPUs
            ipu_dataloader_kwargs_test: Options for the dataloader for the IPU. Ignore if not using IPUs
            args: Arguments for the `DataModule`
            kwargs: Keyword arguments for the `DataModule`
        """
        self.ipu_inference_opts = ipu_inference_opts
        self.ipu_training_opts = ipu_training_opts
        self.ipu_dataloader_training_opts = ipu_dataloader_training_opts
        self.ipu_dataloader_inference_opts = ipu_dataloader_inference_opts

    def _dataloader(self, dataset: Dataset, **kwargs) -> "poptorch.DataLoader":
        """
        Get a poptorch dataloader for a given dataset
        Parameters:
            dataset: The dataset to use
            kwargs: Keyword arguments for the dataloader
        Returns:
            The poptorch dataloader
        """

        # Use regular Dataloader if no IPUs
        if ("ipu_options" not in kwargs.keys()) or (kwargs["ipu_options"] is None):
            raise ValueError(f"No IPU options provided.")

        # Initialize the IPU dataloader
        from goli.ipu.ipu_dataloader import create_ipu_dataloader

        loader = create_ipu_dataloader(
            dataset=dataset,
            **kwargs,
        )

        return loader


class MultitaskFromSmilesDataModule(BaseDataModule, IPUDataModuleModifier):
    def __init__(
        self,
        task_specific_args: Dict[str, Any],  # TODO: Replace this with DatasetParams
        cache_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        featurization_backend: str = "loky",
        featurization_batch_size: int = 1000,
        collate_fn: Optional[Callable] = None,
        prepare_dict_or_graph: str = "pyg:graph",
        dataset_class: type = MultitaskDataset,
        generated_data: bool = False,
        **kwargs,
    ):
        """
        only for parameters beginning with task_*, we have a dictionary where the key is the task name
        and the value is specified below.
        Parameters:
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
            featurization: args to apply to the SMILES to Graph featurizer.
            batch_size_training: batch size for training and val dataset.
            batch_size_inference: batch size for test dataset.
            num_workers: Number of workers for the dataloader. Use -1 to use all available
                cores.
            pin_memory: Whether to pin on paginated CPU memory for the dataloader.
            featurization_n_jobs: Number of cores to use for the featurization.
            featurization_progress: whether to show a progress bar during featurization.
            featurization_backend: The backend to use for the molecular featurization.

                - "multiprocessing": Found to cause less memory issues.
                - "loky": joblib's Default. Found to cause memory leaks.
                - "threading": Found to be slow.
            featurization_batch_size: Batch size to use for the featurization.

            collate_fn: A custom torch collate function. Default is to `goli.data.goli_collate_fn`
            prepare_dict_or_graph: Whether to preprocess all molecules as DGL graphs, DGL dict or PyG graphs.
                Possible options:

                - "dgl:graph": Process molecules as `dgl.DGLGraph`. It's slower during pre-processing
                  and requires more RAM. It is faster during training with `num_workers=0`, but
                  slower with larger `num_workers`.
                - "dgl:dict": Process molecules as a `dict`. It's faster and requires less RAM during
                  pre-processing. It is slower during training with with `num_workers=0` since
                  DGLGraphs will be created during data-loading, but faster with large
                  `num_workers`, and less likely to cause memory issues with the parallelization.
                - "pyg:graph": Process molecules as `pyg.data.Data`.
            dataset_class: The class used to create the dataset from which to sample.
        """
        BaseDataModule.__init__(
            self,
            batch_size_training=batch_size_training,
            batch_size_inference=batch_size_inference,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
        )
        IPUDataModuleModifier.__init__(self, **kwargs)

        self.task_specific_args = task_specific_args
        self.generated_data = generated_data

        # TODO: Have the input argument to the Data Module be of type DatasetParams
        self.task_dataset_processing_params = {
            task: DatasetProcessingParams(**ds_args, generated_data=self.generated_data)
            for task, ds_args in task_specific_args.items()
        }
        self.featurization_n_jobs = featurization_n_jobs
        self.featurization_progress = featurization_progress
        self.featurization_backend = featurization_backend
        self.featurization_batch_size = featurization_batch_size

        self.dataset_class = dataset_class

        self.task_train_indices = None
        self.task_val_indices = None
        self.task_test_indices = None

        self.single_task_datasets = None
        self.train_singletask_datasets = None
        self.val_singletask_datasets = None
        self.test_singletask_datasets = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.cache_data_path = cache_data_path

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

    def generate_data(self, label_cols: List[str], smiles_col: str):
        """
        Line

        Parameters:
            labels_cols
            smiles_col
        Returns:
            pd.DataFrame
        """
        num_generated_mols = int(1e5)
        # Create a dummy generated dataset - singel smiles string, duplicated N times
        # TODO: have both cxsmiles and normal smiles
        # TODO: add the options for different cols for outcomes
        # TODO: Take the label cols etc.
        df = pd.DataFrame(
            [
                dict(
                    cxsmiles="[H]C1C2=C(NC(=O)[C@@]1([H])C1=C([H])C([H])=C(C([H])([H])[H])C([H])=C1[H])C([H])=C([H])N=C2[H] |(6.4528,-1.5789,-1.2859;5.789,-0.835,-0.8455;4.8499,-0.2104,-1.5946;3.9134,0.7241,-0.934;3.9796,1.1019,0.3172;5.0405,0.6404,1.1008;5.2985,1.1457,2.1772;5.9121,-0.5519,0.613;6.9467,-0.2303,0.8014;5.677,-1.7955,1.4745;4.7751,-2.7953,1.0929;4.2336,-2.7113,0.154;4.5521,-3.9001,1.914;3.8445,-4.6636,1.5979;5.215,-4.0391,3.1392;4.9919,-5.2514,4.0126;5.1819,-5.0262,5.0671;5.6619,-6.0746,3.7296;3.966,-5.6247,3.925;6.1051,-3.0257,3.52;6.6247,-3.101,4.4725;6.3372,-1.9217,2.7029;7.0168,-1.1395,3.0281;2.8586,1.2252,-1.7853;2.1303,1.9004,-1.3493;2.8118,0.8707,-3.0956;2.0282,1.2549,-3.7434;3.716,0.0207,-3.7371;4.6658,-0.476,-3.0127;5.3755,-1.1468,-3.5021)|",
                    homo_lumo_gap=1.0,
                )
            ]
        )
        logger.info(f"Generating fake dataset on host... \n Generating {num_generated_mols} rows in the df.")
        df = pd.concat([df] * num_generated_mols, ignore_index=True)
        return df

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

        if self._data_is_prepared:
            logger.info("Data is already prepared. Skipping the preparation")
            return

        # If a path for data caching is provided, try to load from the path.
        # If successful, skip the data preparation.
        cache_data_exists = self.load_data_from_cache()
        if cache_data_exists:
            self._data_is_prepared = True
            return

        """Load all single-task dataframes."""
        task_df = {}
        for task, args in self.task_dataset_processing_params.items():
            logger.info(f"Reading data for task '{task}'")
            if args.generated_data is True:
                # TODO: this is where the generated data should be calculated
                """
                What do we actually need to do here?
                I think it's actually generate the data - if I move it here will we skip the need for the read csv section?
                """
            if args.df is None:
                # Only load the useful columns, as some datasets can be very large when loading all columns.
                label_cols = self._parse_label_cols(
                    df=None, df_path=args.df_path, label_cols=args.label_cols, smiles_col=args.smiles_col
                )
                usecols = (
                    check_arg_iterator(args.smiles_col, enforce_type=list)
                    + label_cols
                    + check_arg_iterator(args.idx_col, enforce_type=list)
                    + check_arg_iterator(args.weights_col, enforce_type=list)
                )
                label_dtype = {col: np.float32 for col in label_cols}
                # TODO: Add an if generated_data here to then go and generate with the appropriate labels etc
                if self.generated_data:
                    task_df[task] = self.generate_data(label_cols=args.label_cols, smiles_col=args.smiles_col)
                else:
                    task_df[task] = self._read_csv(args.df_path, usecols=usecols, dtype=label_dtype)

            else:
                label_cols = self._parse_label_cols(
                    df=args.df, df_path=None, label_cols=args.label_cols, smiles_col=args.smiles_col
                )
                task_df[task] = args.df
            args.label_cols = label_cols
        logger.info("Done reading datasets")

        """Subsample the data frames and extract the necessary data to create SingleTaskDatasets for each task (smiles, labels, extras)."""
        task_dataset_args = {}
        for task in task_df.keys():
            task_dataset_args[task] = {}

        for task, df in task_df.items():
            # Subsample all the dataframes
            sample_size = self.task_dataset_processing_params[task].sample_size
            df = self._sub_sample_df(df, sample_size)

            logger.info(f"Prepare single-task dataset for task '{task}' with {len(df)} data points.")

            # Extract smiles, labels, extras
            args = self.task_dataset_processing_params[task]
            smiles, labels, sample_idx, extras = self._extract_smiles_labels(
                df,
                smiles_col=args.smiles_col,
                label_cols=args.label_cols,
                idx_col=args.idx_col,
                weights_col=args.weights_col,
                weights_type=args.weights_type,
            )

            # Store the relevant information for each task's dataset
            task_dataset_args[task]["smiles"] = smiles
            task_dataset_args[task]["labels"] = labels
            task_dataset_args[task]["sample_idx"] = sample_idx
            task_dataset_args[task]["extras"] = extras

        """Convert SMILES to features (graphs, fingerprints, etc.) for the unique molecules found."""
        all_smiles = []
        all_tasks = []
        idx_per_task = {}
        total_len = 0
        for task, dataset_args in task_dataset_args.items():
            all_smiles.extend(dataset_args["smiles"])
            num_smiles = len(dataset_args["smiles"])
            idx_per_task[task] = (total_len, total_len + num_smiles)
            total_len += num_smiles
            for count in range(len(dataset_args["smiles"])):
                all_tasks.append(task)
        # Get all unique mol ids
        all_mol_ids = smiles_to_unique_mol_ids(
            all_smiles,
            n_jobs=self.featurization_n_jobs,
            featurization_batch_size=self.featurization_batch_size,
            backend=self.featurization_backend,
        )
        # TODO: this needs setting here as well
        if self.generated_data is False:
            # Get all unique mol ids.
            unique_mol_ids, unique_idx, inv = np.unique(all_mol_ids, return_index=True, return_inverse=True)
        else:
            # The generated data is a single molecule duplicated
            mol_ids = np.array(all_mol_ids)
            unique_idx = inv = [_ for _ in range(len(mol_ids))]

        smiles_to_featurize = [all_smiles[ii] for ii in unique_idx]

        # Convert SMILES to features
        features, _ = self._featurize_molecules(
            smiles_to_featurize
        )  # sample_idx is removed ... might need to add it again later in another way

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
                if did_featurization_fail(feat):
                    args["idx_none"].append(idx)

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
                args["extras"],
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
                unique_ids=all_mol_ids[idx_per_task[task][0] : idx_per_task[task][1]],
                **task_dataset_args[task]["extras"],
            )

        """We split the data up to create train, val and test datasets"""
        self.task_train_indices = {}
        self.task_val_indices = {}
        self.task_test_indices = {}

        for task, df in task_df.items():
            train_indices, val_indices, test_indices = self._get_split_indices(
                len(df),
                split_val=self.task_dataset_processing_params[task].split_val,
                split_test=self.task_dataset_processing_params[task].split_test,
                split_seed=self.task_dataset_processing_params[task].split_seed,
                splits_path=self.task_dataset_processing_params[task].splits_path,
                sample_idx=task_dataset_args[task]["sample_idx"],
            )
            self.task_train_indices[task] = train_indices
            self.task_val_indices[task] = val_indices
            self.task_test_indices[task] = test_indices

        (
            self.train_singletask_datasets,
            self.val_singletask_datasets,
            self.test_singletask_datasets,
        ) = self.get_subsets_of_datasets(
            self.single_task_datasets, self.task_train_indices, self.task_val_indices, self.task_test_indices
        )

        # When a path is provided but no cache is found, save to cache
        if (self.cache_data_path is not None) and (not cache_data_exists):
            self.save_data_to_cache()

        self._data_is_prepared = True
        # TODO (Gabriela): Implement the ability to save to cache.

    def setup(
        self,
        stage: str = None,
    ):
        """
        Prepare the torch dataset. Called on every GPUs. Setting state here is ok.
        Parameters:
            stage (str): Either 'fit', 'test', or None.
        """

        # Can possibly get rid of setup because a single dataset will have molecules exclusively in train, val or test
        # Produce the label sizes to update the collate function
        labels_size = {}

        if stage == "fit" or stage is None:
            self.train_ds = MultitaskDataset(self.train_singletask_datasets, n_jobs=self.featurization_n_jobs, backend=self.featurization_backend, featurization_batch_size=self.featurization_batch_size, progress=self.featurization_progress, about="training set", generated=self.generated_data)  # type: ignore
            self.val_ds = MultitaskDataset(self.val_singletask_datasets, n_jobs=self.featurization_n_jobs, backend=self.featurization_backend, featurization_batch_size=self.featurization_batch_size, progress=self.featurization_progress, about="validation set", generated=self.generated_data)  # type: ignore
            print(self.train_ds)
            print(self.val_ds)

            labels_size.update(
                self.train_ds.labels_size
            )  # Make sure that all task label sizes are contained in here. Maybe do the update outside these if statements.
            labels_size.update(self.val_ds.labels_size)

        if stage == "test" or stage is None:
            self.test_ds = MultitaskDataset(self.test_singletask_datasets, n_jobs=self.featurization_n_jobs, backend=self.featurization_backend, featurization_batch_size=self.featurization_batch_size, progress=self.featurization_progress, about="test set", generated=self.generated_data)  # type: ignore
            print(self.test_ds)

            labels_size.update(self.test_ds.labels_size)

        default_labels_size_dict = self.collate_fn.keywords.get("labels_size_dict", None)

        if default_labels_size_dict is None:
            self.collate_fn.keywords["labels_size_dict"] = labels_size

    def get_dataloader_kwargs(self, stage: RunningStage, shuffle: bool, **kwargs) -> Dict[str, Any]:
        """
        Get the options for the dataloader depending on the current stage.

        Parameters:
            stage: Whether in Training, Validating, Testing, Sanity-checking, Predicting, or Tuning phase.
            shuffle: set to ``True`` to have the data reshuffled at every epoch.

        Returns:
            Arguments to pass to the `DataLoader` during initialization
        """
        loader_kwargs = super().get_dataloader_kwargs(stage=stage, shuffle=shuffle, **kwargs)

        # Get batch size and IPU options for training set
        # if stage in [RunningStage.TRAINING, RunningStage.TUNING]:
        if stage in [RunningStage.TRAINING]:
            loader_kwargs["ipu_dataloader_options"] = self.ipu_dataloader_training_opts
            loader_kwargs["ipu_options"] = self.ipu_training_opts

        # Get batch size and IPU options for validation / testing sets
        elif stage in [RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]:
            loader_kwargs["ipu_dataloader_options"] = self.ipu_dataloader_inference_opts
            loader_kwargs["ipu_options"] = self.ipu_inference_opts
        else:
            raise ValueError(f"Wrong value for `stage`. Provided `{stage}`")

        # Remove the IPU options if not available
        if loader_kwargs["ipu_options"] is None:
            loader_kwargs.pop("ipu_options")
            if loader_kwargs["ipu_dataloader_options"] is not None:
                logger.warning(
                    "`ipu_dataloader_options` will be ignored since it is provided without `ipu_options`."
                )
            loader_kwargs.pop("ipu_dataloader_options")
        return loader_kwargs

    def get_dataloader(
        self, dataset: Dataset, shuffle: bool, stage: RunningStage
    ) -> Union[DataLoader, "poptorch.DataLoader"]:
        """
        Get the poptorch dataloader for a given dataset

        Parameters:
            dataset: The dataset from which to load the data
            shuffle: set to ``True`` to have the data reshuffled at every epoch.
            stage: Whether in Training, Validating, Testing, Sanity-checking, Predicting, or Tuning phase.

        Returns:
            The poptorch dataloader to sample from
        """
        kwargs = self.get_dataloader_kwargs(stage=stage, shuffle=shuffle)
        is_ipu = ("ipu_options" in kwargs.keys()) and (kwargs.get("ipu_options") is not None)
        if is_ipu:
            loader = IPUDataModuleModifier._dataloader(self, dataset=dataset, **kwargs)
        else:
            loader = BaseDataModule._dataloader(self, dataset=dataset, **kwargs)

        return loader

    @staticmethod
    def get_collate_fn(collate_fn):
        if collate_fn is None:
            # Some values become `inf` when changing data type. `mask_nan` deals with that
            collate_fn = partial(goli_collate_fn, mask_nan=0, do_not_collate_keys=["smiles", "mol_ids"])
            collate_fn.__name__ = goli_collate_fn.__name__
        return collate_fn

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

        batch_size = BatchingSmilesTransform.parse_batch_size(
            numel=len(smiles),
            desired_batch_size=self.featurization_batch_size,
            n_jobs=self.featurization_n_jobs,
        )

        # Loop all the smiles and compute the features
        features = dm.parallelized_with_batches(
            BatchingSmilesTransform(self.smiles_transformer),
            smiles,
            batch_size=batch_size,
            progress=True,
            n_jobs=self.featurization_n_jobs,
            backend=self.featurization_backend,
            tqdm_kwargs={"desc": f"featurizing_smiles, batch={batch_size}"},
        )

        # Warn about None molecules
        idx_none = [ii for ii, feat in enumerate(features) if did_featurization_fail(feat)]
        if len(idx_none) > 0:
            mols_to_msg = [
                f"idx={idx} - smiles={smiles[idx]} - Error_msg[:-200]=\n{str(features[idx])[:-200]}"
                for idx in idx_none
            ]
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
                    new[key] = MultitaskFromSmilesDataModule._filter_none_molecules(idx_none, val)  # Careful
            else:
                new = arg
            out.append(new)

        out = tuple(out) if len(out) > 1 else out[0]

        return out

    def _parse_label_cols(
        self,
        df: pd.DataFrame,
        df_path: Optional[Union[str, os.PathLike]],
        label_cols: Union[Type[None], str, List[str]],
        smiles_col: str,
    ) -> List[str]:
        r"""
        Parse the choice of label columns depending on the type of input.
        The input parameters `label_cols` and `smiles_col` are described in
        the `__init__` method.
        Parameters:
            df: The dataframe containing the labels.
            df_path: The path to the dataframe containing the labels.
            label_cols: The columns to use as labels.
            smiles_col: The column to use as SMILES
        Returns:
            the parsed label columns
        """
        if df is None:
            # Only load the useful columns, as some dataset can be very large
            # when loading all columns
            # TODO: Option if generated_data and no df make the df with the relevant cols
            # TODO: or we could just return the relevant cols
            if self.generated_data:
                data_frame = self.generate_data(label_cols=label_cols, smiles_col=smiles_col)
            else:
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
        num_feats = graph.feat.shape[1]
        return num_feats

    @property
    def in_dims(self):
        """
        Return all input dimensions for the set of graphs.
        Including node/edge features, and
        raw positional encoding dimensions such eigval, eigvec, rwse and more
        """

        graph = self.get_first_graph()
        if isinstance(graph, (dgl.DGLGraph, GraphDict)):
            graph = graph.ndata

        # get list of all keys corresponding to positional encoding
        pe_dim_dict = {}
        g_keys = graph.keys

        # ignore the normal keys for node feat and edge feat etc.
        for key in g_keys:
            prop = graph.get(key, None)
            if hasattr(prop, "shape"):
                pe_dim_dict[key] = prop.shape[-1]
        return pe_dim_dict

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
        error. If all 20 first element, then `None` is returned, otherwise the first
        graph to not fail is returned.
        """
        keys = list(self.task_dataset_processing_params.keys())
        task = keys[0]
        args = self.task_dataset_processing_params[task]
        if args.df is None:
            # TODO: Again if there is generated data flag and no df yet, need to make it here
            if self.generated_data:
                df = self.generate_data(label_cols=args.label_cols, smiles_col=args.smiles_col)
            else:
                df = self._read_csv(args.df_path, nrows=20)
        else:
            df = args.df.iloc[0:20, :]

        label_cols = self._parse_label_cols(
            df, df_path=None, label_cols=args.label_cols, smiles_col=args.smiles_col
        )

        smiles, labels, sample_idx, extras = self._extract_smiles_labels(
            df,
            smiles_col=args.smiles_col,
            label_cols=label_cols,
            idx_col=args.idx_col,
            weights_col=args.weights_col,
            weights_type=args.weights_type,
        )

        graph = None
        for s in smiles:
            graph = self.smiles_transformer(s, mask_nan=0.0)
            num_nodes = get_num_nodes(graph)
            num_edges = get_num_edges(graph)
            if (graph is not None) and (num_edges > 0) and (num_nodes > 0):
                break
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
        """
        For a given dataframe extract the SMILES and labels columns. Smiles is returned as a list
        of string while labels are returned as a 2D numpy array.

        Parameters:
            df: Pandas dataframe
            smiles_col: Name of the column containing the SMILES
            label_cols: List of column names containing the labels
            idx_col: Name of the column containing the index
            weights_col: Name of the column containing the weights
            weights_type: Type of weights to use.
        Returns:
            smiles, labels, sample_idx, extras
        """

        if smiles_col is None:  # Should we specify which dataset has caused the potential issue?
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

        indices = None  # What are indices for?
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
        r"""
        Compute indices of random splits.
        Parameters:
            dataset_size: Size of the dataset
            split_val: Fraction of the dataset to use for validation
            split_test: Fraction of the dataset to use for testing
            sample_idx: Indices of the samples to use for splitting
            split_seed: Seed for the random splitting
            splits_path: Path to a file containing the splits
        Returns:
            train_indices, val_indices, test_indices
        """

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
                # TODO: Don't need the generated data here - maybe have an assertation line or something
                if self.generated_data:
                    splits = self.generate_data()
                else:
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

    def _sub_sample_df(self, df: pd.DataFrame, sample_size: Union[int, float, None]) -> pd.DataFrame:
        r"""
        subsample from a pandas dataframe
        Parameters:
            df: pandas dataframe to subsample
            sample_size: number of samples to subsample
        Returns:
            subsampled pandas dataframe
        """
        # Sub-sample the dataframe
        if isinstance(sample_size, int):
            n = min(sample_size, df.shape[0])
            df = df.sample(n=n)
        elif isinstance(sample_size, float):
            df = df.sample(f=sample_size)
        elif sample_size is None:
            pass
        else:
            raise ValueError(
                f"Wrong value for `sample_size`: {sample_size}"
            )  # Maybe specify which task it was for?

        return df

    def get_data_hash(self):
        """
        Get a hash specific to a dataset and smiles_transformer.
        Useful to cache the pre-processed data.
        """
        hash_dict = {
            "smiles_transformer": self.smiles_transformer,
            "task_specific_args": self.task_specific_args,
        }
        data_hash = get_md5_hash(hash_dict)
        return data_hash

    def get_data_cache_fullname(self, compress: bool = False) -> str:
        """
        Create a hash for the dataset, and use it to generate a file name

        Parameters:
            compress: Whether to compress the data
        Returns:
            full path to the data cache file
        """
        if self.cache_data_path is None:
            return
        data_hash = self.get_data_hash()
        ext = ".datacache"
        if compress:
            ext += ".gz"
        data_cache_fullname = fs.join(self.cache_data_path, data_hash + ext)
        return data_cache_fullname

    def save_data_to_cache(self, verbose: bool = True, compress: bool = False) -> None:
        """
        Save the datasets from cache. First create a hash for the dataset, use it to
        generate a file name. Then save to the path given by `self.cache_data_path`.

        Parameters:
            verbose: Whether to print the progress
            compress: Whether to compress the data

        """
        full_cache_data_path = self.get_data_cache_fullname(compress=compress)
        if full_cache_data_path is None:
            logger.info("No cache data path specified. Skipping saving the data to cache.")
            return

        save_params = {
            "single_task_datasets": self.single_task_datasets,
            "task_train_indices": self.task_train_indices,
            "task_val_indices": self.task_val_indices,
            "task_test_indices": self.task_test_indices,
        }

        fs.mkdir(self.cache_data_path)
        with fsspec.open(full_cache_data_path, mode="wb", compression="infer") as file:
            if verbose:
                logger.info(f"Saving the data to cache at path:\n`{full_cache_data_path}`")
            now = time.time()
            torch.save(save_params, file)
            elapsed = round(time.time() - now)
            if verbose:
                logger.info(
                    f"Successfully saved the data to cache in {elapsed}s at path: `{full_cache_data_path}`"
                )

    def load_data_from_cache(self, verbose: bool = True, compress: bool = False) -> bool:
        """
        Load the datasets from cache. First create a hash for the dataset, and verify if that
        hash is available at the path given by `self.cache_data_path`.

        Parameters:
            verbose: Whether to print the progress
            compress: Whether to compress the data

        Returns:
            cache_data_exists: Whether the cache exists (if the hash matches) and the loading succeeded
        """
        full_cache_data_path = self.get_data_cache_fullname(compress=compress)

        if full_cache_data_path is None:
            logger.info("No cache data path specified. Skipping loading the data from cache.")
            return False

        cache_data_exists = fs.exists(full_cache_data_path)

        if cache_data_exists:
            try:
                logger.info(f"Loading the data from cache at path `{full_cache_data_path}`")
                now = time.time()
                with fsspec.open(full_cache_data_path, mode="rb", compression="infer") as file:
                    load_params = torch.load(file)
                    self.__dict__.update(load_params)
                    (
                        self.train_singletask_datasets,
                        self.val_singletask_datasets,
                        self.test_singletask_datasets,
                    ) = self.get_subsets_of_datasets(
                        self.single_task_datasets,
                        self.task_train_indices,
                        self.task_val_indices,
                        self.task_test_indices,
                    )
                elapsed = round(time.time() - now)
                logger.info(
                    f"Successfully loaded the data from cache in {elapsed}s at path: `{full_cache_data_path}`"
                )
                return True
            except Exception as e:
                if verbose:
                    logger.warning(
                        f"Data cache failed to load path: `{full_cache_data_path}`.\nThe data will be prepared and cache will be created for future runs."
                    )
                    logger.warning(e.__str__())
                return False
        else:
            if verbose:
                logger.info(
                    f"Data cache not found at path: `{full_cache_data_path}`.\nThe data will be prepared and cache will be created for future runs."
                )
            return False

    def get_subsets_of_datasets(
        self,
        single_task_datasets: Dict[str, SingleTaskDataset],
        task_train_indices: Dict[str, Iterable],
        task_val_indices: Dict[str, Iterable],
        task_test_indices: Dict[str, Iterable],
    ) -> Tuple[Subset, Subset, Subset]:
        """
        From a dictionary of datasets and their associated indices, subset the train/val/test sets

        Parameters:
            single_task_datasets: Dictionary of datasets
            task_train_indices: Dictionary of train indices
            task_val_indices: Dictionary of val indices
            task_test_indices: Dictionary of test indices
        Returns:
            train_singletask_datasets: Dictionary of train subsets
            val_singletask_datasets: Dictionary of val subsets
            test_singletask_datasets: Dictionary of test subsets
        """
        train_singletask_datasets = {}
        val_singletask_datasets = {}
        test_singletask_datasets = {}
        for task in task_train_indices.keys():
            train_singletask_datasets[task] = Subset(single_task_datasets[task], task_train_indices[task])
            val_singletask_datasets[task] = Subset(single_task_datasets[task], task_val_indices[task])
            test_singletask_datasets[task] = Subset(single_task_datasets[task], task_test_indices[task])
        return train_singletask_datasets, val_singletask_datasets, test_singletask_datasets

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current DataModule, which is the combined size of all single-task datasets given.
        Returns:
            num_elements: Number of elements in the current DataModule
        """
        num_elements = 0
        for task, args in self.task_dataset_processing_params.items():
            if args.df is None:
                # TODO: The full generated data here if they don't yet exist.
                if self.generated_data:
                    df = self.generate_data(label_cols=args.label_cols, smiles_col=args.smiles_col)
                else:
                    df = self._read_csv(args.df_path, usecols=[args.smiles_col])
                num_elements += len(df)
            else:
                num_elements += len(args.df)
        return num_elements

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the current DataModule
        Returns:
            obj_repr: Dictionary representation of the current DataModule
        """
        # TODO: Change to make more multi-task friendly
        obj_repr = {}
        obj_repr["name"] = self.__class__.__name__
        obj_repr["len"] = len(self)
        obj_repr["train_size"] = len(self.train_indices) if self.train_indices is not None else None
        obj_repr["val_size"] = len(self.val_indices) if self.val_indices is not None else None
        obj_repr["test_size"] = len(self.test_indices) if self.test_indices is not None else None
        obj_repr["batch_size_training"] = self.batch_size_training
        obj_repr["batch_size_inference"] = self.batch_size_inference
        obj_repr["num_node_feats"] = self.num_node_feats
        obj_repr["num_node_feats_with_positional_encoding"] = self.num_node_feats_with_positional_encoding
        obj_repr["num_edge_feats"] = self.num_edge_feats
        obj_repr["num_tasks"] = len(self.task_dataset_processing_params)
        obj_repr["num_labels"] = len(self.label_cols)
        obj_repr["collate_fn"] = self.collate_fn.__name__
        obj_repr["featurization"] = self.featurization
        return obj_repr

    def __repr__(self) -> str:
        r"""
        Controls how the class is printed

        Returns:

        """
        return omegaconf.OmegaConf.to_yaml(self.to_dict())


class GraphOGBDataModule(MultitaskFromSmilesDataModule):
    def __init__(
        self,
        task_specific_args: Dict[str, Dict[str, Any]],  # TODO: Replace this with DatasetParams
        cache_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        featurization_n_jobs: int = -1,
        featurization_progress: bool = False,
        featurization_backend: str = "loky",
        collate_fn: Optional[Callable] = None,
        prepare_dict_or_graph: str = "pyg:graph",
        dataset_class: type = MultitaskDataset,
        **kwargs,
    ):
        r"""
        Load an OGB (Open-graph-benchmark) GraphProp dataset.

        Parameters:
            task_specific_args: Arguments related to each task, with the task-name being the key,
              and the specific arguments being the values. The arguments must be a
              Dict containing the following keys:

              - "dataset_name": Name of the OGB dataset to load. Examples of possible datasets are
                "ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molfreesolv".
              - "sample_size": The number of molecules to sample from the dataset. Default=None,
                meaning that all molecules will be considered.
            cache_data_path: path where to save or reload the cached data. The path can be
                remote (S3, GS, etc).
            featurization: args to apply to the SMILES to Graph featurizer.
            batch_size_training: batch size for training and val dataset.
            batch_size_inference: batch size for test dataset.
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

        new_task_specific_args = {}
        self.metadata = {}
        for task_name, task_args in task_specific_args.items():
            # Get OGB metadata
            this_metadata = self._get_dataset_metadata(task_args["dataset_name"])
            # Get dataset
            df, idx_col, smiles_col, label_cols, splits_path = self._load_dataset(
                this_metadata, sample_size=task_args.get("sample_size", None)
            )
            new_task_specific_args[task_name] = {
                "df": df,
                "idx_col": idx_col,
                "smiles_col": smiles_col,
                "label_cols": label_cols,
                "splits_path": splits_path,
            }
            self.metadata[task_name] = this_metadata

        # Config for datamodule
        dm_args = {}
        dm_args["task_specific_args"] = new_task_specific_args
        dm_args["cache_data_path"] = cache_data_path
        dm_args["featurization"] = featurization
        dm_args["batch_size_training"] = batch_size_training
        dm_args["batch_size_inference"] = batch_size_inference
        dm_args["num_workers"] = num_workers
        dm_args["pin_memory"] = pin_memory
        dm_args["featurization_n_jobs"] = featurization_n_jobs
        dm_args["featurization_progress"] = featurization_progress
        dm_args["featurization_backend"] = featurization_backend
        dm_args["persistent_workers"] = persistent_workers
        dm_args["collate_fn"] = collate_fn
        dm_args["prepare_dict_or_graph"] = prepare_dict_or_graph
        dm_args["dataset_class"] = dataset_class

        super().__init__(**dm_args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        r"""
        geenrate a dictionary representation of the class
        Returns:
            dict: dictionary representation of the class
        """
        # TODO: Change to make more multi-task friendly
        obj_repr = {}
        obj_repr["dataset_name"] = self.dataset_name
        obj_repr.update(super().to_dict())
        return obj_repr

    # Private methods

    def _load_dataset(
        self,
        metadata: dict,
        sample_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, str, str, List[str], str]:
        """
        Download, extract and load an OGB dataset.
        Parameters:
            metadata: Metadata for the dataset to load.
            sample_size: The number of molecules to sample from the dataset. Default=None,
                meaning that all molecules will be considered.
        Returns:
            df: Pandas dataframe containing the dataset.
            idx_col: Name of the column containing the molecule index.
            smiles_col: Name of the column containing the SMILES.
            label_cols: List of column names containing the labels.
            splits_path: Path to the file containing the train/val/test splits.
        """

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

        # Subsample the dataset
        df = self._sub_sample_df(df, sample_size)

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

    def _get_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        ogb_metadata = self._get_ogb_metadata()
        if dataset_name not in ogb_metadata.index:
            raise ValueError(f"'{dataset_name}' is not a valid dataset name.")

        return ogb_metadata.loc[dataset_name].to_dict()

    def _get_ogb_metadata(self):
        """
        Get the metadata of OGB GraphProp datasets.
        """

        with importlib.resources.open_text("ogb.graphproppred", "master.csv") as f:
            ogb_metadata = pd.read_csv(f)
        ogb_metadata = ogb_metadata.set_index(ogb_metadata.columns[0])
        ogb_metadata = ogb_metadata.T

        # Add metadata related to PCQM4M
        ogb_metadata = pd.concat([ogb_metadata, pd.DataFrame(PCQM4M_meta, index=["ogbg-lsc-pcqm4m"])])
        ogb_metadata = pd.concat([ogb_metadata, pd.DataFrame(PCQM4Mv2_meta, index=["ogbg-lsc-pcqm4mv2"])])

        # Only keep datasets of type 'mol'
        ogb_metadata = ogb_metadata[ogb_metadata["data type"] == "mol"]

        return ogb_metadata


def get_num_nodes(
    graph: Union[dgl.DGLGraph, GraphDict, Data, Batch],
) -> int:
    """
    utility function to get the number of nodes in a graph
    Parameters:
        graph: a DGLGraph, GraphDict, Data or Batch object
    Returns:
        num_nodes: the number of nodes in the graph
    """
    if isinstance(graph, (dgl.DGLGraph, GraphDict)):
        return graph.num_nodes()
    elif isinstance(graph, (Data, Batch)):
        return graph.num_nodes
    else:
        raise ValueError(f"graph dtype not recognised.")


def get_num_edges(
    graph: Union[dgl.DGLGraph, GraphDict, Data, Batch],
) -> int:
    """
    Utility function to get the number of edges in a graph
    Parameters:
        graph: a DGLGraph, GraphDict, Data or Batch object
    Returns:
        num_edges: the number of edges in the graph
    """
    if isinstance(graph, (dgl.DGLGraph, GraphDict)):
        return graph.num_edges()
    elif isinstance(graph, (Data, Batch)):
        return graph.num_edges
