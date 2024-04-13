"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals, Graphcore Limited, and NVIDIA Corporation & Affiliates are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""


import os
from copy import deepcopy
from functools import lru_cache
from multiprocessing import Manager
from typing import Any, Dict, List, Optional, Tuple, Union
from collections.abc import Callable

import fsspec
import numpy as np
import torch
from datamol import parallelized, parallelized_with_batches
from loguru import logger
from torch.utils.data.dataloader import Dataset
from torch_geometric.data import Batch, Data

from graphium.features import GraphDict

import graphium_cpp


class SingleTaskDataset(Dataset):
    def __init__(
        self,
        labels: List[Union[torch.Tensor, np.ndarray]],
        features: Optional[List[Union[Data, "GraphDict"]]] = None,
        smiles: Optional[List[str]] = None,
        indices: Optional[List[int]] = None,
        weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
        unique_ids: Optional[List[str]] = None,
        mol_ids: Optional[List[str]] = None,
    ):
        r"""
        dataset for a single task
        Parameters:
            labels: A list of labels for the given task (one per graph)
            features: A list of graphs
            smiles: A list of smiles
            indices: A list of indices
            weights: A list of weights
            unique_ids: A list of unique ids for each molecule generated from `datamol.unique_id`
            mol_ids: A list of ids coming from the original dataset. Useful to identify the molecule in the original dataset.
        """

        # Verify that all lists are the same length
        numel = len(labels)

        def _check_if_same_length(to_check, label):
            """Simple utility method to throw an error if the length is not as expected."""
            if to_check is not None and len(to_check) != numel:
                raise ValueError(
                    f"{label} must be the same length as `labels`, got {len(to_check)} and {numel}"
                )

        _check_if_same_length(features, "features")
        _check_if_same_length(indices, "indices")
        _check_if_same_length(weights, "weights")
        _check_if_same_length(unique_ids, "unique_ids")
        _check_if_same_length(mol_ids, "mol_ids")

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
        self.mol_ids = mol_ids

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

        if self.mol_ids is not None:
            datum["mol_ids"] = self.mol_ids[idx]

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
        state["mol_ids"] = self.mol_ids
        return state

    def __setstate__(self, state: dict):
        """Reload the class from pickling."""
        if state["smiles"] is not None:
            manager = Manager()
            state["smiles"] = manager.list(state["smiles"])

        self.__dict__.update(state)


class MultitaskDataset(Dataset):
    pass

    def __init__(
        self,
        featurize_smiles: Callable[[str],dict],
        task_names: List[str],
        label_num_cols: List[int],
        label_dtypes: List[int],
        mol_file_data_offsets,
        concat_smiles_tensor,
        smiles_offsets_tensor,
        num_nodes_tensor,
        num_edges_tensor,
        about: str = "",
        data_path: Optional[Union[str, os.PathLike]] = None,
    ):
        r"""
        This class holds the information for the multitask dataset.
        we will have a multitask dataset of the following form:
        - self.mol_file_data_offsets will be a Tensor representing where to find
            label data about each molecule in the corresponding file
        - self.smiles_tensor will be a Tensor containing all smiles strings concatenated, with null terminators
        - self.smiles_offsets_tensor will be a Tensor indicating where smiles strings start in smiles_tensor
        - self.num_nodes_tensor will be a Tensor of the number of nodes in each graph
        - self.num_edges_tensor will be a Tensor of the number of edges in each graph

        Parameters:
            about: A description of the dataset
            data_path: The location of the data if saved on disk
        """
        super().__init__()

        self.about = about
        self.data_path = data_path
        self.featurize_smiles = featurize_smiles
        self.task_names = task_names
        self.label_num_cols = label_num_cols
        self.label_dtypes = label_dtypes
        self.mol_file_data_offsets = mol_file_data_offsets
        self.smiles_tensor = concat_smiles_tensor
        self.smiles_offsets_tensor = smiles_offsets_tensor
        self.num_nodes_tensor = num_nodes_tensor
        self.num_edges_tensor = num_edges_tensor
        self.dataset_length = num_nodes_tensor.size(dim=0)

        logger.info(f"Dataloading from DISK")

    def __len__(self):
        r"""
        Returns the number of molecules
        """
        return self.dataset_length

    @property
    def num_nodes_list(self):
        """
        The number of nodes per graph
        """
        return self.num_nodes_tensor

    @property
    def num_edges_list(self):
        """
        The number of edges per graph
        """
        return self.num_edges_tensor

    @property
    def num_graphs_total(self):
        r"""
        number of graphs (molecules) in the dataset
        """
        return len(self)

    @property
    def num_nodes_total(self):
        """Total number of nodes for all graphs"""
        if len(self) == 0:
            return
        return torch.sum(self.num_nodes_list, dtype=torch.int64).item()

    @property
    def max_num_nodes_per_graph(self):
        """Maximum number of nodes per graph"""
        if len(self) == 0:
            return
        return torch.max(self.num_nodes_list).item()

    @property
    def std_num_nodes_per_graph(self):
        """Standard deviation of number of nodes per graph"""
        if len(self) == 0:
            return
        # correction is zero to match previous default behaviour of numpy.std
        # Consider changing it to 1 (the torch.std default)
        return torch.std(self.num_nodes_list.to(torch.float64), correction=0).item()

    @property
    def min_num_nodes_per_graph(self):
        """Minimum number of nodes per graph"""
        if len(self) == 0:
            return
        return torch.min(self.num_nodes_list).item()

    @property
    def mean_num_nodes_per_graph(self):
        """Average number of nodes per graph"""
        if len(self) == 0:
            return
        return self.num_nodes_total / self.num_graphs_total

    @property
    def num_edges_total(self):
        """Total number of edges for all graphs"""
        if len(self) == 0:
            return
        return torch.sum(self.num_edges_list, dtype=torch.int64).item()

    @property
    def max_num_edges_per_graph(self):
        """Maximum number of edges per graph"""
        if len(self) == 0:
            return
        return torch.max(self.num_edges_list).item()

    @property
    def min_num_edges_per_graph(self):
        """Minimum number of edges per graph"""
        if len(self) == 0:
            return
        return torch.min(self.num_edges_list).item()

    @property
    def std_num_edges_per_graph(self):
        """Standard deviation of number of nodes per graph"""
        if len(self) == 0:
            return
        # correction is zero to match previous default behaviour of numpy.std
        # Consider changing it to 1 (the torch.std default)
        return torch.std(self.num_edges_list.to(torch.float64), correction=0).item()

    @property
    def mean_num_edges_per_graph(self):
        """Average number of edges per graph"""
        if len(self) == 0:
            return
        return self.num_edges_total / self.num_graphs_total

    def __getitem__(self, idx):
        r"""
        get the data for at the specified index
        Parameters:
            idx: The index of the data to retrieve
        Returns:
            A dictionary containing the data for the specified index with keys "labels", "num_nodes", "num_edges", and "features"
        """
        if self.smiles_tensor is None or self.smiles_offsets_tensor is None:
            raise ValueError("Missing smiles in MultitaskDataset.__getitem__")

        smiles_str = graphium_cpp.extract_string(self.smiles_tensor, self.smiles_offsets_tensor, idx)

        datum = {
            "labels": self.load_graph_from_index(idx),
            "features": self.featurize_smiles(smiles_str),
        }

        # One of the featurization error handling options returns a string on error,
        # instead of throwing an exception, so assume that the intention is to just skip,
        # instead of crashing.
        if isinstance(datum["features"], str):
            datum = None

        return datum

    def load_graph_from_index(self, data_idx):
        r"""
        load the graph (in pickle file) from the disk
        Parameters:
            data_idx: The index of the data to retrieve
        Returns:
            A Data object containing the data for the specified index with keys corresponding to the tasks.
        """
        labels = {}
        graphium_cpp.load_labels_from_index(self.data_path, data_idx, self.mol_file_data_offsets, self.task_names, self.label_num_cols, self.label_dtypes, labels)
        data_dict = Data()
        for task, values in labels.items():
            data_dict[task] = values

        return data_dict

    def __repr__(self) -> str:
        """
        summarizes the dataset in a string
        Returns:
            A string representation of the dataset.
        """
        if len(self) == 0:
            out_str = (
                f"-------------------\n{self.__class__.__name__}\n"
                + f"\tabout = {self.about}\n"
                + f"\tnum_graphs_total = {self.num_graphs_total}\n"
                + f"-------------------\n"
            )
            return out_str

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


class FakeDataset(MultitaskDataset):
    """
    A dataset to hold the fake data.
    """

    def __init__(
        self, datasets: Dict[str, SingleTaskDataset], num_mols: int = 1234, indexing_same_elem: bool = False
    ):
        """
        Parameters:
            datasets:
                A dictionary of datasets. The keys are the task names and the values are the datasets.
            num_mols:
                The number of molecules to generate. In reality, it is the same molecule,
                but `num_mols` will change the length of the dataset.
            indexing_same_elem:
                If True, the same molecule is used for all samples.
                Otherwise, a deepcopied molecule is used for each sample.
        """
        self.indexing_same_elem = indexing_same_elem
        self.num_mols = num_mols
        self.num_datasets = len(datasets)

        self.about = "FakeDatasets"
        task = next(iter(datasets))
        self.features = None
        self.labels = None
        if "features" in datasets[task][0]:
            self.mol_ids, self.smiles, self.labels, self.features = self.merge(datasets)
            if self.indexing_same_elem is False:
                self.mol_ids, self.smiles, self.labels, self.features = self.deepcopy_mol(
                    self.mol_ids, self.smiles, self.labels, self.features
                )
        else:
            self.mol_ids, self.smiles, self.labels = self.merge(datasets)
            if self.indexing_same_elem is False:
                self.mol_ids, self.smiles, self.labels, _ = self.deepcopy_mol(
                    self.mol_ids, self.smiles, self.labels
                )

        self.label_num_cols = self.set_label_num_cols(datasets)
        self.label_dtypes = self.set_label_dtype_dict(datasets)

    def _get_inv_of_mol_ids(self, all_mol_ids):
        # The generated data is a single molecule duplicated
        mol_ids = np.array(all_mol_ids)
        inv = [_ for _ in range(len(mol_ids) // self.num_datasets)] * self.num_datasets
        mol_ids = np.unique(inv)
        return mol_ids, inv

    def deepcopy_mol(self, mol_ids, labels, smiles, features=None):
        """
        Create a deepcopy of the single molecule num_mols times

        Args:
            mol_ids (array): The single value for the mol ID
            labels (List[Dict]): List containing one dict with the label name-value pairs
            smiles (List[List[str]]): List of list containing SMILE sting
            features (List[Data], optional): list containing Data object. Defaults to None.

        Returns:
            The deep copy of the inputs
        """
        logger.info("Duplicating the single dataset element...")
        mol_ids = [deepcopy(mol_ids[0]) for _ in range(self.num_mols)]
        logger.info("Finished `mol_ids`")
        labels = [deepcopy(labels[0]) for _ in range(self.num_mols)]
        logger.info("Finished `labels`")
        smiles = [deepcopy(smiles[0]) for _ in range(self.num_mols)]
        logger.info("Finished `smiles`")
        if features is not None:
            features = [deepcopy(features[0]) for _ in range(self.num_mols)]
            logger.info("Finished `features`")
        return mol_ids, labels, smiles, features

    def __len__(self):
        r"""
        Returns the number of molecules
        """
        return self.num_mols

    def __getitem__(self, idx):
        r"""
        get the data for at the specified index
        Parameters:
            idx: The index of the data to retrieve
        Returns:
            A dictionary containing the data for the specified index with keys "mol_ids", "smiles", "labels", and "features"
        """
        datum = {}
        if self.indexing_same_elem is True:
            # If using a single memory location override the idx value passed
            idx = 0
        if self.labels is not None:
            datum["labels"] = self.labels[idx]

        if self.features is not None:
            datum["features"] = self.features[idx]

        return datum

def torch_enum_to_dtype(v: Union[int, torch.dtype]):
    if isinstance(v, torch.dtype):
        return v

    mapping = [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex32,
        torch.complex64,
        torch.complex128,
        torch.bool,
        torch.qint8,
        torch.quint8,
        torch.qint32,
        torch.bfloat16,
        torch.quint4x2
    ]
    return mapping[v] if (v >= 0 and v < len(mapping)) else None

def get_num_nodes_per_graph(graphs):
    r"""
    number of nodes per graph
    """
    if isinstance(graphs, Batch):
        graphs = graphs.to_data_list()
    counts = [graph.num_nodes for graph in graphs]
    return counts


def get_num_edges_per_graph(graphs):
    r"""
    number of edges per graph
    """
    if isinstance(graphs, Batch):
        graphs = graphs.to_data_list()
    counts = [graph.num_edges for graph in graphs]
    return counts
