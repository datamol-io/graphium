import os
from copy import deepcopy
from functools import lru_cache
from multiprocessing import Manager
from typing import Any, Dict, List, Optional, Tuple, Union

import fsspec
import numpy as np
import torch
from datamol import parallelized, parallelized_with_batches
from loguru import logger
from torch.utils.data.dataloader import Dataset
from torch_geometric.data import Batch, Data

from graphium.data.smiles_transform import smiles_to_unique_mol_ids
from graphium.features import GraphDict


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
        datasets: Dict[str, SingleTaskDataset],
        n_jobs=-1,
        backend: str = "loky",
        featurization_batch_size=1000,
        progress: bool = True,
        save_smiles_and_ids: bool = False,
        about: str = "",
        data_path: Optional[Union[str, os.PathLike]] = None,
        dataloading_from: str = "ram",
        data_is_cached: bool = False,
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
            featurization_batch_size: The batch size to use for the parallelization of the featurization
            progress: Whether to display the progress bar
            save_smiles_and_ids: Whether to save the smiles and ids for the dataset. If `False`, `mol_ids` and `smiles` are set to `None`
            about: A description of the dataset
            data_path: The location of the data if saved on disk
            dataloading_from: Whether to load the data from `"disk"` or `"ram"`
            data_is_cached: Whether the data is already cached on `"disk"`
        """
        super().__init__()
        self.n_jobs = n_jobs
        self.backend = backend
        self.featurization_batch_size = featurization_batch_size
        self.progress = progress
        self.about = about
        self.save_smiles_and_ids = save_smiles_and_ids
        self.data_path = data_path
        self.dataloading_from = dataloading_from

        logger.info(f"Dataloading from {dataloading_from.upper()}")

        if data_is_cached:
            self._load_metadata()

            if dataloading_from == "disk":
                self.features = None
                self.labels = None
            elif dataloading_from == "ram":
                logger.info(f"Transferring {about} from DISK to RAM...")
                self.transfer_from_disk_to_ram()

        else:
            task = next(iter(datasets))
            self.features = None
            if (len(datasets[task]) > 0) and ("features" in datasets[task][0]):
                self.mol_ids, self.smiles, self.labels, self.features = self.merge(datasets)
            else:
                self.mol_ids, self.smiles, self.labels = self.merge(datasets)
            # Set mol_ids and smiles to None to save memory as they are not needed.
            if not save_smiles_and_ids:
                self.mol_ids = None
                self.smiles = None
            self.labels_size = self.set_label_size_dict(datasets)
            self.labels_dtype = self.set_label_dtype_dict(datasets)
            self.dataset_length = len(self.labels)
            self._num_nodes_list = None
            self._num_edges_list = None
            if self.features is not None:
                self._num_nodes_list = get_num_nodes_per_graph(self.features)
                self._num_edges_list = get_num_edges_per_graph(self.features)

    def transfer_from_disk_to_ram(self, parallel_with_batches: bool = False):
        """
        Function parallelizing transfer from DISK to RAM
        """

        def transfer_mol_from_disk_to_ram(idx):
            """
            Function transferring single mol from DISK to RAM
            """
            data_dict = self.load_graph_from_index(idx)
            mol_in_ram = {
                "features": data_dict["graph_with_features"],
                "labels": data_dict["labels"],
            }

            return mol_in_ram

        if parallel_with_batches and self.featurization_batch_size:
            data_in_ram = parallelized_with_batches(
                transfer_mol_from_disk_to_ram,
                range(self.dataset_length),
                batch_size=self.featurization_batch_size,
                n_jobs=0,
                backend=self.backend,
                progress=self.progress,
                tqdm_kwargs={"desc": "Transfer from DISK to RAM"},
            )
        else:
            data_in_ram = parallelized(
                transfer_mol_from_disk_to_ram,
                range(self.dataset_length),
                n_jobs=0,
                backend=self.backend,
                progress=self.progress,
                tqdm_kwargs={"desc": "Transfer from DISK to RAM"},
            )

        self.features = [sample["features"] for sample in data_in_ram]
        self.labels = [sample["labels"] for sample in data_in_ram]

    def save_metadata(self, directory: str):
        """
        Save everything other than features/labels
        """
        attrs_to_save = [
            "mol_ids",
            "smiles",
            "labels_size",
            "labels_dtype",
            "dataset_length",
            "_num_nodes_list",
            "_num_edges_list",
        ]
        attrs = {attr: getattr(self, attr) for attr in attrs_to_save}

        path = os.path.join(directory, "multitask_metadata.pkl")

        torch.save(attrs, path, pickle_protocol=4)

    def _load_metadata(self):
        """
        Load everything other than features/labels
        """
        attrs_to_load = [
            "mol_ids",
            "smiles",
            "labels_size",
            "labels_dtype",
            "dataset_length",
            "_num_nodes_list",
            "_num_edges_list",
        ]
        path = os.path.join(self.data_path, "multitask_metadata.pkl")
        with fsspec.open(path, "rb") as f:
            attrs = torch.load(path)

        if not set(attrs_to_load).issubset(set(attrs.keys())):
            raise ValueError(
                f"The metadata in the cache at {self.data_path} does not contain the right information. "
                f"This may be because the cache was prepared using an earlier version of Graphium. "
                f"You can try deleting the cache and running the data preparation again. "
                f"\nMetadata keys found: {attrs.keys()}"
                f"\nMetadata keys required: {attrs_to_load}"
            )

        for attr, value in attrs.items():
            setattr(self, attr, value)

        if self.save_smiles_and_ids:
            if self.smiles is None or self.mol_ids is None:
                logger.warning(
                    f"Argument `save_smiles_and_ids` is set to {self.save_smiles_and_ids} but metadata in the cache at {self.data_path} does not contain smiles and mol_ids. "
                    f"This may be because `Datamodule.prepare_data(save_smiles_and_ids=False)` was run followed by `Datamodule.setup(save_smiles_and_ids=True)`. "
                    f"When loading from cached files, the `save_smiles_and_ids` argument of `Datamodule.setup()` is superseeded by the `Datamodule.prepare_data()`. "
                )

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
        if self._num_nodes_list is None:
            if len(self) == 0:
                self._num_nodes_list = []
            else:
                self._num_nodes_list = get_num_nodes_per_graph(self.features)
        return self._num_nodes_list

    @property
    def num_edges_list(self):
        """
        The number of edges per graph
        """
        if self._num_edges_list is None:
            if len(self) == 0:
                self._num_edges_list = []
            else:
                self._num_edges_list = get_num_edges_per_graph(self.features)
        return self._num_edges_list

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
        return sum(self.num_nodes_list)

    @property
    def max_num_nodes_per_graph(self):
        """Maximum number of nodes per graph"""
        if len(self) == 0:
            return
        return max(self.num_nodes_list)

    @property
    def std_num_nodes_per_graph(self):
        """Standard deviation of number of nodes per graph"""
        if len(self) == 0:
            return
        return np.std(self.num_nodes_list)

    @property
    def min_num_nodes_per_graph(self):
        """Minimum number of nodes per graph"""
        if len(self) == 0:
            return
        return min(self.num_nodes_list)

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
        return sum(self.num_edges_list)

    @property
    def max_num_edges_per_graph(self):
        """Maximum number of edges per graph"""
        if len(self) == 0:
            return
        return max(self.num_edges_list)

    @property
    def min_num_edges_per_graph(self):
        """Minimum number of edges per graph"""
        if len(self) == 0:
            return
        return min(self.num_edges_list)

    @property
    def std_num_edges_per_graph(self):
        """Standard deviation of number of nodes per graph"""
        if len(self) == 0:
            return
        return np.std(self.num_edges_list)

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
            A dictionary containing the data for the specified index with keys "mol_ids", "smiles", "labels", and "features"
        """
        datum = {}
        if self.dataloading_from == "disk":
            data_dict = self.load_graph_from_index(idx)
            datum["features"] = data_dict["graph_with_features"]
            datum["labels"] = data_dict["labels"]
            if "smiles" in data_dict.keys():
                datum["smiles"] = data_dict["smiles"]
        else:
            if self.mol_ids is not None:
                datum["mol_ids"] = self.mol_ids[idx]

            if self.smiles is not None:
                datum["smiles"] = self.smiles[idx]

            if self.labels is not None:
                datum["labels"] = self.labels[idx]

            if self.features is not None:
                datum["features"] = self.features[idx]

        return datum

    def load_graph_from_index(self, data_idx):
        r"""
        load the graph (in pickle file) from the disk
        Parameters:
            data_idx: The index of the data to retrieve
        Returns:
            A dictionary containing the data for the specified index with keys "graph_with_features", "labels" and "smiles" (optional).
        """
        filename = os.path.join(
            self.data_path, format(data_idx // 1000, "04d"), format(data_idx, "07d") + ".pkl"
        )
        with fsspec.open(filename, "rb") as f:
            data_dict = torch.load(f)
        return data_dict

    def merge(
        self, datasets: Dict[str, SingleTaskDataset]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[Any]]:
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

        # Get all the smiles, labels, features and tasks.
        all_lists = self._get_all_lists_ids(datasets=datasets)
        mol_ids, inv = self._get_inv_of_mol_ids(all_mol_ids=all_lists["mol_ids"])

        # Store the smiles.
        smiles = [[] for _ in range(len(mol_ids))]
        for all_idx, unique_idx in enumerate(inv):
            smiles[unique_idx].append(all_lists["smiles"][all_idx])

        # Store the labels.
        labels = [Data() for _ in range(len(mol_ids))]
        for all_idx, unique_idx in enumerate(inv):
            task: str = all_lists["tasks"][all_idx]
            label = all_lists["labels"][all_idx]
            labels[unique_idx][task] = label

            if all_idx < len(all_lists["features"]):
                features = all_lists["features"][all_idx]
                labels[unique_idx]["x"] = torch.empty(
                    (features.num_nodes, 1)
                )  # IPU is not happy with zero-sized tensors, so use shape (features.num_nodes, 1) here
                labels[unique_idx]["edge_index"] = torch.empty((2, features.num_edges))

        # Store the features
        if len(all_lists["features"]) > 0:
            features = [-1 for i in range(len(mol_ids))]
            for all_idx, unique_idx in enumerate(inv):
                features[unique_idx] = all_lists["features"][all_idx]
            return mol_ids, smiles, labels, features
        else:
            return mol_ids, smiles, labels

    def _get_all_lists_ids(self, datasets: Dict[str, SingleTaskDataset]) -> Dict[str, Any]:
        all_smiles = []
        all_features = []
        all_labels = []
        all_mol_ids = []
        all_tasks = []

        for task, ds in datasets.items():
            if len(ds) == 0:
                continue
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

        all_lists = {
            "smiles": all_smiles,
            "features": all_features,
            "labels": all_labels,
            "mol_ids": all_mol_ids,
            "tasks": all_tasks,
        }

        return all_lists

    def _get_inv_of_mol_ids(self, all_mol_ids):
        mol_ids, inv = np.unique(all_mol_ids, return_inverse=True)
        return mol_ids, inv

    def _find_valid_label(self, task, ds):
        r"""
        For a given dataset, find a genuine label for that dataset
        """
        valid_label = None
        for i in range(len(ds)):
            if ds[i] is not None:
                valid_label = ds[i]["labels"]
                break

        if valid_label is None:
            raise ValueError(f"Dataset for task {task} has no valid labels.")

        return valid_label

    def set_label_size_dict(self, datasets: Dict[str, SingleTaskDataset]):
        r"""
        This gives the number of labels to predict for a given task.
        """
        task_labels_size = {}
        for task, ds in datasets.items():
            if len(ds) == 0:
                continue

            valid_label = self._find_valid_label(task, ds)

            # Assume for a fixed task, the label dimension is the same across data points
            torch_label = torch.as_tensor(valid_label)

            # First dimension is graph-specific
            task_labels_size[task] = torch_label.size()
        return task_labels_size

    def set_label_dtype_dict(self, datasets: Dict[str, SingleTaskDataset]):
        r"""
        Gets correct dtype for a given label
        """
        task_labels_dtype = {}
        for task, ds in datasets.items():
            if len(ds) == 0:
                continue

            valid_label = self._find_valid_label(task, ds)

            torch_label = torch.as_tensor(valid_label)
            task_labels_dtype[task] = torch_label.dtype
        return task_labels_dtype

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

        # Faster to compute the statistics if we unbatch first.
        features = self.features
        if isinstance(self.features, Batch):
            self.features = self.features.to_data_list()

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

        # Restore the original features.
        self.features = features

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

        self.labels_size = self.set_label_size_dict(datasets)
        self.labels_dtype = self.set_label_dtype_dict(datasets)
        self.features = self.features

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
