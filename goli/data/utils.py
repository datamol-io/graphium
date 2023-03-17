import importlib.resources
import zipfile

import pandas as pd

import goli

GOLI_DATASETS_BASE_URL = "gcs://goli-public/datasets"
GOLI_DATASETS = {
    "goli-zinc-micro": "goli-zinc-micro.zip",
    "goli-zinc-bench-gnn": "goli-zinc-bench-gnn.zip",
    "goli-htsfp": "goli-htsfp.csv.gz",
}


def load_micro_zinc() -> pd.DataFrame:
    """
    Return a dataframe of micro ZINC (1000 data points).
    Returns:
        pd.DataFrame: A dataframe of micro ZINC.
    """

    with importlib.resources.open_text("goli.data.micro_ZINC", "micro_ZINC.csv") as f:
        df = pd.read_csv(f)

    return df  # type: ignore


def load_tiny_zinc() -> pd.DataFrame:
    """
    Return a dataframe of tiny ZINC (100 data points).
    Returns:
        pd.DataFrame: A dataframe of tiny ZINC.
    """

    with importlib.resources.open_text("goli.data.micro_ZINC", "micro_ZINC.csv") as f:
        df = pd.read_csv(f, nrows=100)

    return df  # type: ignore


def list_goli_datasets() -> set:
    """
    List Goli datasets available to download.
    Returns:
        set: A set of Goli dataset names.
    """
    return set(GOLI_DATASETS.keys())


def download_goli_dataset(
    name: str, output_path: str, extract_zip: bool = True, progress: bool = False
) -> str:
    r"""Download a Goli dataset to a specified location.

    Args:
        name: Name of the Goli dataset from `goli.data.utils.get_goli_datasets()`.
        output_path: Directory path where to download the dataset to.
        extract_zip: Whether to extract the dataset if it's a zip file.
        progress: Whether to show a progress bar during download.

    Returns:
        str: Path to the downloaded dataset.
    """

    if name not in GOLI_DATASETS:
        raise ValueError(f"'{name}' is not a valid Goli dataset name. Choose from {GOLI_DATASETS}")

    fname = GOLI_DATASETS[name]

    dataset_path_source = goli.utils.fs.join(GOLI_DATASETS_BASE_URL, fname)
    dataset_path_destination = goli.utils.fs.join(output_path, fname)

    if not goli.utils.fs.exists(dataset_path_destination):
        goli.utils.fs.copy(dataset_path_source, dataset_path_destination, progress=progress)

        if extract_zip and str(dataset_path_destination).endswith(".zip"):
            # Unzip the dataset
            with zipfile.ZipFile(dataset_path_destination, "r") as zf:
                zf.extractall(output_path)

    if extract_zip:
        # Set the destination path to the folder
        # NOTE(hadim): this is a bit fragile.
        dataset_path_destination = dataset_path_destination.split(".")[0]

    return dataset_path_destination
