from typing import Union, List, Callable, Dict, Tuple, Any, Optional

import importlib.resources
import zipfile

from loguru import logger

import pandas as pd
import numpy as np

import graphium

from torch_geometric.data import Data
from graphium.features.featurizer import GraphDict

GRAPHIUM_DATASETS_BASE_URL = "gs://graphium-public/datasets"
GRAPHIUM_DATASETS = {
    "graphium-zinc-micro": "zinc-micro.zip",
    "graphium-zinc-bench-gnn": "zinc-bench-gnn.zip",
}


def load_micro_zinc() -> pd.DataFrame:
    """
    Return a dataframe of micro ZINC (1000 data points).
    Returns:
        pd.DataFrame: A dataframe of micro ZINC.
    """

    with importlib.resources.open_text("graphium.data.micro_ZINC", "micro_ZINC.csv") as f:
        df = pd.read_csv(f)

    return df  # type: ignore


def load_tiny_zinc() -> pd.DataFrame:
    """
    Return a dataframe of tiny ZINC (100 data points).
    Returns:
        pd.DataFrame: A dataframe of tiny ZINC.
    """

    with importlib.resources.open_text("graphium.data.micro_ZINC", "micro_ZINC.csv") as f:
        df = pd.read_csv(f, nrows=100)

    return df  # type: ignore


def graphium_package_path(graphium_path: str) -> str:
    """Return the path of a graphium file in the package."""

    assert graphium_path.startswith(
        "graphium://"
    ), f"Invalid graphium path, must start with 'graphium://': {graphium_path}"

    graphium_path = graphium_path.replace("graphium://", "")
    package, ressource = graphium_path.split("/")
    with importlib.resources.path(package, ressource) as data_path:
        pass
    return str(data_path)


def list_graphium_datasets() -> set:
    """
    List Graphium datasets available to download.
    Returns:
        set: A set of Graphium dataset names.
    """
    return set(GRAPHIUM_DATASETS.keys())


def download_graphium_dataset(
    name: str, output_path: str, extract_zip: bool = True, progress: bool = False
) -> str:
    r"""Download a Graphium dataset to a specified location.

    Args:
        name: Name of the Graphium dataset from `graphium.data.utils.get_graphium_datasets()`.
        output_path: Directory path where to download the dataset to.
        extract_zip: Whether to extract the dataset if it's a zip file.
        progress: Whether to show a progress bar during download.

    Returns:
        str: Path to the downloaded dataset.
    """

    if name not in GRAPHIUM_DATASETS:
        raise ValueError(f"'{name}' is not a valid Graphium dataset name. Choose from {GRAPHIUM_DATASETS}")

    fname = GRAPHIUM_DATASETS[name]

    dataset_path_source = graphium.utils.fs.join(GRAPHIUM_DATASETS_BASE_URL, fname)
    dataset_path_destination = graphium.utils.fs.join(output_path, fname)

    if not graphium.utils.fs.exists(dataset_path_destination):
        graphium.utils.fs.copy(dataset_path_source, dataset_path_destination, progress=progress)

        if extract_zip and str(dataset_path_destination).endswith(".zip"):
            # Unzip the dataset
            with zipfile.ZipFile(dataset_path_destination, "r") as zf:
                zf.extractall(output_path)

    if extract_zip:
        # Set the destination path to the folder
        # NOTE(hadim): this is a bit fragile.
        dataset_path_destination = dataset_path_destination.split(".")[0]

    return dataset_path_destination


def get_keys(pyg_data):
    if isinstance(type(pyg_data).keys, property):
        return pyg_data.keys
    else:
        return pyg_data.keys()


def found_size_mismatch(task: str, features: Union[Data, GraphDict], labels: np.ndarray, smiles: str) -> bool:
    """Check if a size mismatch exists between features and labels with respect to node/edge/nodepair.

    Args:
        task: The task name is needed to determine the task level (graph, node, edge or nodepair)
        features: Features/information of molecule/graph (e.g., edge_index, feat, edge_feat, num_nodes, etc.)
        labels: Target label of molecule for the task
        smiles: Smiles string of molecule

    Returns:
        mismatch: Boolean variable indicating if a size mismatch was found between featurs and labels.
    """

    mismatch = False

    if np.isnan(labels).any():
        pass

    elif task.startswith("graph_"):
        pass

    elif task.startswith("node_"):
        if labels.shape[0] != features.num_nodes:
            mismatch = True
            logger.warning(
                (
                    f"Inconsistent number of nodes between labels and features in {task} task for {smiles}: {labels.shape[0]} vs {features.num_nodes}"
                )
            )

    elif task.startswith("edge_"):
        if labels.shape[0] != features.num_edges:
            mismatch = True
            logger.warning(
                (
                    f"Inconsistent number of edges between labels and features in {task} task for {smiles}: {labels.shape[0]} vs {features.num_edges}"
                )
            )

    elif task.startswith("nodepair_"):
        if list(labels.shape[:2]) != 2 * [features.num_nodes]:
            mismatch = True
            logger.warning(
                (
                    f"Inconsistent shape of nodepairs between labels and features in {task} task for {smiles}: {list(labels.shape[:2])} vs {2 * [features.num_nodes]}"
                )
            )

    else:
        raise ValueError("Unkown task level")

    return mismatch
