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

import tempfile
from contextlib import redirect_stderr, redirect_stdout
from typing import Type, List, Dict, Union, Any, Callable, Optional, Tuple, Iterable, Literal
from os import PathLike as Path

from dataclasses import dataclass

import os
from functools import partial
import importlib.resources
import zipfile
from copy import deepcopy
import time
import gc

import platformdirs
import re
from graphium.data.utils import get_keys

from loguru import logger
import fsspec
import omegaconf

import pandas as pd
import numpy as np
import datamol as dm
from tqdm import tqdm
import os.path as osp
from fastparquet import ParquetFile

from sklearn.model_selection import train_test_split

import lightning
from lightning.pytorch.trainer.states import RunningStage

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from torch.utils.data import Subset

from rdkit import RDLogger

from graphium.utils import fs
from graphium.features import mol_to_pyggraph

from graphium.data.sampler import DatasetSubSampler
from graphium.data.utils import graphium_package_path, found_size_mismatch
from graphium.utils.arg_checker import check_arg_iterator
from graphium.utils.hashing import get_md5_hash
from graphium.data.collate import graphium_collate_fn
import graphium.data.dataset as Datasets
from graphium.data.normalization import LabelNormalization
from graphium.data.multilevel_utils import extract_labels

import graphium_cpp

torch.multiprocessing.set_sharing_strategy("file_system")


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


class BaseDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        batch_size_per_pack: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        multiprocessing_context: Optional[str] = None,
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
            multiprocessing_context: multiprocessing context for data worker creation
            collate_fn: collate function for batching
        """
        super().__init__()

        self.batch_size_training = batch_size_training
        self.batch_size_inference = batch_size_inference
        self.batch_size_per_pack = batch_size_per_pack
        if self.batch_size_per_pack is not None:
            # Check that batch_size_per_pack is a divisor of batch_size_training and batch_size_inference
            assert (
                self.batch_size_training % self.batch_size_per_pack == 0
            ), f"batch_size_training must be a multiple of batch_size_per_pack, provided batch_size_training={self.batch_size_training}, batch_size_per_pack={self.batch_size_per_pack}"
            assert (
                self.batch_size_inference % self.batch_size_per_pack == 0
            ), f"batch_size_inference must be a multiple of batch_size_per_pack, provided batch_size_inference={self.batch_size_inference}, batch_size_per_pack={self.batch_size_per_pack}"

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.multiprocessing_context = multiprocessing_context

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

    def get_collate_fn(self, collate_fn):
        if collate_fn is None:
            # Some values become `inf` when changing data type. `mask_nan` deals with that
            collate_fn = partial(
                graphium_collate_fn, mask_nan=0, batch_size_per_pack=self.batch_size_per_pack
            )
            collate_fn.__name__ = graphium_collate_fn.__name__

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

    def get_fake_graph(self):
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

        path = str(path)

        if path.endswith((".csv", ".csv.gz", ".csv.zip", ".csv.bz2")):
            sep = ","
        elif path.endswith((".tsv", ".tsv.gz", ".tsv.zip", ".tsv.bz2")):
            sep = "\t"
        else:
            raise ValueError(f"unsupported file `{path}`")
        kwargs.setdefault("sep", sep)

        if path.startswith("graphium://"):
            path = graphium_package_path(path)

        df = pd.read_csv(path, **kwargs)
        return df

    @staticmethod
    def _get_data_file_type(path):
        # Extract the extension
        name, ext = os.path.splitext(path)

        # support compressed files
        _, ext2 = os.path.splitext(name)
        if ext2 != "":
            ext = f"{ext2}{ext}"

        if ext.endswith((".parquet")):  # Support parquet files. Compression is implicit
            return "parquet"
        elif ".sdf" in ext:  # support compressed sdf files
            return "sdf"
        elif ".csv" in ext:  # support compressed csv files
            return "csv"
        elif ".tsv" in ext:  # support compressed tsv files
            return "tsv"
        elif ".pkl" in ext:  # support compressed pickle files
            return "pkl"
        elif ext.endswith(".pt"):  # Pytorch tensor files, used for storing index splits
            return "pt"
        else:
            raise ValueError(f"unsupported file `{path}`")

    @staticmethod
    def _get_table_columns(path: str) -> List[str]:
        """
        Get the columns of a table without reading all the data.
        Might be slow to decompress the file if the file is compressed.

        Parameters:
            path: path to the table file

        Returns:
            List[str]: the column names
        """

        datafile_type = BaseDataModule._get_data_file_type(path)

        if datafile_type == "parquet":
            # Read the schema of a parquet file
            file = ParquetFile(path)
            schema = file.pandas_metadata["columns"]
            column_names = [s["name"] for s in schema if s["name"] is not None]
        elif datafile_type == "sdf":
            df = BaseDataModule._read_sdf(path, max_num_mols=5, discard_invalud=True, n_jobs=1)
            column_names = df.columns
        elif datafile_type in ["csv", "tsv"]:
            # Read the schema of a csv / tsv file
            df = BaseDataModule._read_csv(path, nrows=5)
            column_names = df.columns
        return column_names

    @staticmethod
    def _read_parquet(path, **kwargs):
        kwargs.pop("dtype", None)  # Only useful for csv
        column_names = BaseDataModule._get_table_columns(path)

        # Change the 'usecols' parameter to 'columns'
        columns = kwargs.pop("columns", None)
        if "usecols" in kwargs.keys():
            assert columns is None, "Ambiguous value of `columns`"
            columns = kwargs.pop("usecols")
        if columns is None:
            columns = column_names
        for column in columns:
            assert (
                column in column_names
            ), f"Column `{column}` is not in the parquet file with columns {column_names}"

        # Read the parquet file per column, and convert the data to float16 to reduce memory consumption
        all_series = {}
        progress = tqdm(columns)
        for col in progress:
            # Read single column
            progress.set_description(f"Reading parquet column `{col}`")
            this_series = pd.read_parquet(path, columns=[col], engine="fastparquet", **kwargs)[col]

            # Check if the data is float
            first_elem = this_series.values[0]
            is_float = False
            if isinstance(first_elem, (list, tuple)):
                is_float = isinstance(first_elem[0], float) or (first_elem[0].dtype.kind == "f")
            elif isinstance(first_elem, np.ndarray):
                is_float = isinstance(first_elem, float) or (first_elem.dtype.kind == "f")

            # Convert floats to float16
            if is_float:
                if isinstance(first_elem, (np.ndarray, list)):
                    this_series.update(
                        pd.Series([np.asarray(elem).astype(np.float16) for elem in this_series])
                    )
                else:
                    this_series = this_series.astype(np.float16)

            all_series[col] = this_series
            gc.collect()  # Reset memory after each column

        # Merge columns into a dataframe
        df = pd.concat(all_series, axis=1)

        return df

    @staticmethod
    def _read_sdf(path: str, mol_col_name: str = "_rdkit_molecule_obj", **kwargs):
        r"""
        read a given sdf file into a pandas dataframe
        uses datamol.read_sdf to read the sdf file
        Parameters:
            path: path to the sdf file
            mol_col_name: name of the column containing the molecule object
            kwargs: arguments to pass to datamol.read_sdf
        Returns:
            pandas dataframe containing the molecules and their conformer coordinates from the sdf file

        Note: please change mol_col_name to be the column name corresoonding to the molecule object in your sdf file
        """

        # Set default arguments for reading the SDF
        kwargs.setdefault("smiles_column", "smiles")
        kwargs.setdefault("sanitize", False)
        kwargs.setdefault("include_private", True)
        kwargs.setdefault("include_computed", True)
        kwargs.setdefault("remove_hs", True)
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("max_num_mols", kwargs.pop("sample_size", None))
        kwargs.setdefault("discard_invalid", False)

        # Get the interesting columns
        mol_cols = mol_col_name  # adjust this to be the column name in your sdf file
        kwargs.setdefault("mol_column", mol_cols)
        usecols = kwargs.pop("usecols", None)
        dtype = kwargs.pop("dtype", None)
        smiles_col = kwargs["smiles_column"]

        # Read the SDF
        df = dm.read_sdf(path, as_df=True, **kwargs)

        # Keep only the columns needed
        if usecols is not None:
            df = df[usecols + [mol_cols]]

        # Convert the dtypes
        if dtype is not None:
            label_columns = list(set(usecols) - set([mol_cols, smiles_col]))
            dtype_mapper = {col: dtype for col in label_columns}
            df.astype(dtype=dtype_mapper, copy=False)

        return df

    @staticmethod
    def _glob(path: str) -> List[str]:
        """
        glob a given path
        Parameters:
            path: path to glob
        Returns:
            List[str]: list of paths
        """
        files = dm.fs.glob(path)
        files = [f.replace("file://", "") for f in files]
        return files

    def _read_table(self, path: str, **kwargs) -> pd.DataFrame:
        """
        a general read file function which determines if which function to use, either _read_csv or _read_parquet
        Parameters:
            path: path to the file to read
            kwargs: keyword arguments for pd.read_csv or pd.read_parquet
        Returns:
            pd.DataFrame: the panda dataframe storing molecules
        """
        files = self._glob(path)
        if len(files) == 0:
            raise FileNotFoundError(f"No such file or directory `{path}`")

        if len(files) > 1:
            files = tqdm(sorted(files), desc=f"Reading files at `{path}`")
        dfs = []
        for file in files:
            file_type = self._get_data_file_type(file)
            if file_type == "parquet":
                df = self._read_parquet(file, **kwargs)
            elif file_type == "sdf":  # support compressed sdf files
                df = self._read_sdf(file, **kwargs)
            elif file_type in ["csv", "tsv"]:  # support compressed csv and tsv files
                df = self._read_csv(file, **kwargs)
            else:
                raise ValueError(f"unsupported file `{file}`")
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

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
        loader_kwargs["multiprocessing_context"] = self.multiprocessing_context

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
            logger.info("Max num nodes being calcuated train")
            max_num_nodes = max(max_num_nodes, self.train_ds.max_num_nodes_per_graph)

        # Max number of nodes in the validation dataset
        if (
            (self.val_ds is not None)
            and ("val" in stages)
            and (self.val_ds.max_num_nodes_per_graph is not None)
        ):
            logger.info("Max num nodes being calcuated val")
            max_num_nodes = max(max_num_nodes, self.val_ds.max_num_nodes_per_graph)

        # Max number of nodes in the test dataset
        if (
            (self.test_ds is not None)
            and ("test" in stages)
            and (self.test_ds.max_num_nodes_per_graph is not None)
        ):
            logger.info("Max num nodes being calcuated test")
            max_num_nodes = max(max_num_nodes, self.test_ds.max_num_nodes_per_graph)

        # Max number of nodes in the predict dataset
        if (
            (self.predict_ds is not None)
            and ("predict" in stages)
            and (self.predict_ds.max_num_nodes_per_graph is not None)
        ):
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
        if (
            (self.train_ds is not None)
            and ("train" in stages)
            and (self.train_ds.max_num_edges_per_graph is not None)
        ):
            max_num_edges = max(max_num_edges, self.train_ds.max_num_edges_per_graph)

        # Max number of nodes/edges in the validation dataset
        if (
            (self.val_ds is not None)
            and ("val" in stages)
            and (self.val_ds.max_num_edges_per_graph is not None)
        ):
            max_num_edges = max(max_num_edges, self.val_ds.max_num_edges_per_graph)

        # Max number of nodes/edges in the test dataset
        if (
            (self.test_ds is not None)
            and ("test" in stages)
            and (self.test_ds.max_num_edges_per_graph is not None)
        ):
            max_num_edges = max(max_num_edges, self.test_ds.max_num_edges_per_graph)

        # Max number of nodes/edges in the predict dataset
        if (
            (self.predict_ds is not None)
            and ("predict" in stages)
            and (self.predict_ds.max_num_edges_per_graph is not None)
        ):
            max_num_edges = max(max_num_edges, self.predict_ds.max_num_edges_per_graph)

        return max_num_edges


@dataclass
class DatasetProcessingParams:
    def __init__(
        self,
        task_level: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        df_path: Optional[Union[str, os.PathLike, List[Union[str, os.PathLike]]]] = None,
        smiles_col: Optional[str] = None,
        label_cols: List[str] = None,
        weights_col: Optional[str] = None,  # Not needed
        weights_type: Optional[str] = None,  # Not needed
        idx_col: Optional[str] = None,
        mol_ids_col: Optional[str] = None,
        sample_size: Union[int, float, Type[None]] = None,
        split_val: float = 0.2,
        split_test: float = 0.2,
        seed: int = None,
        epoch_sampling_fraction: float = 1.0,
        splits_path: Optional[Union[str, os.PathLike]] = None,
        split_names: Optional[List[str]] = ["train", "val", "test"],
        label_normalization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
    ):
        """
        object to store the parameters for the dataset processing
        Parameters:
            task_level: The task level, wether it is graph, node, edge or nodepair
            df: The dataframe containing the data
            df_path: The path to the dataframe containing the data. If list, will read all files, sort them alphabetically and concatenate them.
            smiles_col: The column name of the smiles
            label_cols: The column names of the labels
            weights_col: The column name of the weights
            weights_type: The type of weights
            idx_col: The column name of the indices
            mol_ids_col: The column name of the molecule ids
            sample_size: The size of the sample
            split_val: The fraction of the data to use for validation
            split_test: The fraction of the data to use for testing
            seed: The seed to use for the splits and subsampling
            splits_path: The path to the splits, or a dictionary with the splits
        """

        if df is None and df_path is None:
            raise ValueError("Either `df` or `df_path` must be provided")
        if epoch_sampling_fraction <= 0 or epoch_sampling_fraction > 1:
            raise ValueError("The value of epoch_sampling_fraction must be in the range of (0, 1].")

        self.df = df
        self.task_level = task_level
        self.df_path = df_path
        self.smiles_col = smiles_col
        self.label_cols = label_cols
        self.weights_col = weights_col
        self.weights_type = weights_type
        self.idx_col = idx_col
        self.mol_ids_col = mol_ids_col
        self.sample_size = sample_size
        self.split_val = split_val
        self.split_test = split_test
        self.seed = seed
        self.splits_path = splits_path
        self.split_names = split_names
        self.label_normalization = label_normalization
        self.epoch_sampling_fraction = epoch_sampling_fraction


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
        from graphium.ipu.ipu_dataloader import create_ipu_dataloader

        loader = create_ipu_dataloader(
            dataset=dataset,
            **kwargs,
        )

        return loader


class MultitaskFromSmilesDataModule(BaseDataModule, IPUDataModuleModifier):
    def __init__(
        self,
        task_specific_args: Union[Dict[str, DatasetProcessingParams], Dict[str, Any]],
        processed_graph_data_path: Union[str, os.PathLike],
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        batch_size_per_pack: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        multiprocessing_context: Optional[str] = None,
        collate_fn: Optional[Callable] = None,
        preprocessing_n_jobs: int = 0,
        **kwargs,
    ):
        """
        only for parameters beginning with task_*, we have a dictionary where the key is the task name
        and the value is specified below.
        Parameters:
            task_specific_args: A dictionary where the key is the task name (for the multi-task setting), and
                the value is a `DatasetProcessingParams` object. The `DatasetProcessingParams` object
                contains multiple parameters to define how to load and process the files, such as:

                - `task_level`
                - `df`
                - `df_path`
                - `smiles_col`
                - `label_cols`
            featurization: args to apply to the SMILES to Graph featurizer.
            batch_size_training: batch size for training and val dataset.
            batch_size_inference: batch size for test dataset.
            num_workers: Number of workers for the dataloader. Use -1 to use all available
                cores.
            pin_memory: Whether to pin on paginated CPU memory for the dataloader.

                - "multiprocessing": Found to cause less memory issues.
                - "loky": joblib's Default. Found to cause memory leaks.
                - "threading": Found to be slow.

            collate_fn: A custom torch collate function. Default is to `graphium.data.graphium_collate_fn`
            preprocessing_n_jobs: Number of threads to use during preprocessing.
                Use 0 to use all available cores, or -1 to use all but one core.

            dataloading_from: Deprecated. Behaviour now always matches previous "disk" option.
            featurization_n_jobs: Deprecated.
            featurization_progress: Deprecated.
            featurization_backend: Deprecated.
            featurization_batch_size: Deprecated.
            prepare_dict_or_graph: Deprecated. Behaviour now always matches previous "pyg:graph" option.
        """
        BaseDataModule.__init__(
            self,
            batch_size_training=batch_size_training,
            batch_size_inference=batch_size_inference,
            batch_size_per_pack=batch_size_per_pack,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            collate_fn=collate_fn,
        )
        IPUDataModuleModifier.__init__(self, **kwargs)

        self.task_specific_args = task_specific_args

        self.task_dataset_processing_params = {}
        for task, ds_args in task_specific_args.items():
            if not isinstance(ds_args, DatasetProcessingParams):
                # This is needed as long as not all classes have been migrated
                # to use the new `DatasetProcessingParams` class
                ds_args = DatasetProcessingParams(**ds_args)

            key = self._get_task_key(ds_args.task_level, task)
            self.task_dataset_processing_params[key] = ds_args

        self.sampler_task_dict = {
            task: self.task_dataset_processing_params[task].epoch_sampling_fraction
            for task in self.task_dataset_processing_params.keys()
        }
        self.task_names = [task for task in self.task_dataset_processing_params.keys()]

        self.task_train_indices = None
        self.task_val_indices = None
        self.task_test_indices = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self._parse_caching_args(processed_graph_data_path)

        self.task_norms = {}

        if featurization is None:
            featurization = {}

        self.featurization = featurization

        # Copy featurization for the representation used by graphium_cpp
        encoded_featurization = deepcopy(featurization)
        self.encoded_featurization = encoded_featurization

        def encode_feature_options(options, name, encoding_function):
            if name not in options or options[name] is None:
                options[name] = torch.tensor(data=[], dtype=torch.int64)
            else:
                options[name] = encoding_function(options[name])

        encode_feature_options(
            encoded_featurization,
            "atom_property_list_onehot",
            graphium_cpp.atom_onehot_feature_names_to_tensor,
        )
        encode_feature_options(
            encoded_featurization, "atom_property_list_float", graphium_cpp.atom_float_feature_names_to_tensor
        )
        encode_feature_options(
            encoded_featurization, "edge_property_list", graphium_cpp.bond_feature_names_to_tensor
        )

        if (
            "pos_encoding_as_features" in featurization
            and featurization["pos_encoding_as_features"] is not None
            and featurization["pos_encoding_as_features"]["pos_types"] is not None
        ):
            (pos_encoding_names, pos_encoding_tensor) = graphium_cpp.positional_feature_options_to_tensor(
                featurization["pos_encoding_as_features"]["pos_types"]
            )
        else:
            pos_encoding_names = []
            pos_encoding_tensor = torch.tensor(data=[], dtype=torch.int64)
        encoded_featurization["pos_encoding_as_features"] = (pos_encoding_names, pos_encoding_tensor)

        explicit_H = featurization["explicit_H"] if "explicit_H" in featurization else False
        add_self_loop = featurization["add_self_loop"] if "add_self_loop" in featurization else False
        merge_equivalent_mols = (
            featurization["merge_equivalent_mols"] if "merge_equivalent_mols" in featurization else True
        )

        # Save these for calling graphium_cpp.prepare_and_save_data later
        self.add_self_loop = add_self_loop
        self.explicit_H = explicit_H
        self.merge_equivalent_mols = merge_equivalent_mols

        self.preprocessing_n_jobs = preprocessing_n_jobs

        self.smiles_transformer = partial(mol_to_pyggraph, **encoded_featurization)
        self.data_hash = self.get_data_hash()

        if self._ready_to_load_all_from_file():
            self._data_is_prepared = True

    def _parse_caching_args(self, processed_graph_data_path):
        """
        Parse the caching arguments, and raise errors if the arguments are invalid.
        """

        # If loading from disk, the path to the cached data must be provided
        if processed_graph_data_path is None:
            raise ValueError("`processed_graph_data_path` must be provided.")

        self.processed_graph_data_path = processed_graph_data_path

    def _get_task_key(self, task_level: str, task: str):
        task_prefix = f"{task_level}_"
        if not task.startswith(task_prefix):
            task = task_prefix + task
        return task

    def get_task_levels(self):
        task_level_map = {}

        for task, task_args in self.task_specific_args.items():
            if isinstance(task_args, DatasetProcessingParams):
                task_args = task_args.__dict__  # Convert the class to a dictionary
            task_level_map.update({task: task_args["task_level"]})

        return task_level_map

    @staticmethod
    def concat_smiles_tensor_index():
        return 0

    @staticmethod
    def smiles_offsets_tensor_index():
        return 1

    @staticmethod
    def num_nodes_tensor_index():
        return 2

    @staticmethod
    def num_edges_tensor_index():
        return 3

    @staticmethod
    def data_offsets_tensor_index():
        return 4

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
            - Split the dataset according to the task-specific splits for train, val and test
        """

        # Don't log error messages from SMILES parsing in RDKit.
        # Common error messages were:
        # WARNING: not removing hydrogen atom without neighbors
        # SMILES Parse Error: syntax error while parsing: restricted
        # SMILES Parse Error: Failed parsing SMILES 'restricted' for input: 'restricted'
        RDLogger.DisableLog("rdApp.*")

        for task, args in self.task_dataset_processing_params.items():
            if args.label_normalization is None:
                args.label_normalization = {}
            label_normalization = LabelNormalization(**args.label_normalization)
            self.task_norms[task] = label_normalization

        if self._data_is_prepared:
            logger.info("Data is already prepared.")
            self.label_num_cols, self.label_dtypes = graphium_cpp.load_num_cols_and_dtypes(
                self.processed_graph_data_path, self.data_hash
            )
            self.stage_data = {
                "train": graphium_cpp.load_metadata_tensors(
                    self.processed_graph_data_path, "train", self.data_hash
                ),
                "val": graphium_cpp.load_metadata_tensors(
                    self.processed_graph_data_path, "val", self.data_hash
                ),
                "test": graphium_cpp.load_metadata_tensors(
                    self.processed_graph_data_path, "test", self.data_hash
                ),
            }
            if len(self.label_num_cols) > 0:
                for task in self.task_dataset_processing_params.keys():
                    stats = graphium_cpp.load_stats(self.processed_graph_data_path, self.data_hash, task)
                    if len(stats) < 4:
                        raise RuntimeError(f'Error loading cached stats for task "{task}"')

                    self.task_norms[task].set_statistics(stats[0], stats[1], stats[2], stats[3])
            return

        task_dataset_args = {}
        self.task_train_indices = {}
        self.task_val_indices = {}
        self.task_test_indices = {}

        """Load all single-task dataframes."""
        for task, args in self.task_dataset_processing_params.items():
            logger.info(f"Reading data for task '{task}'")
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
                df = self._read_table(args.df_path, usecols=usecols, dtype=label_dtype)

            else:
                label_cols = self._parse_label_cols(
                    df=args.df, df_path=None, label_cols=args.label_cols, smiles_col=args.smiles_col
                )
                df = args.df

            args.label_cols = label_cols

            """Subsample the data frames and extract the necessary data for each task (smiles, labels, extras)."""

            # Subsample all the dataframes
            sample_size = self.task_dataset_processing_params[task].sample_size
            df = self._sub_sample_df(df, sample_size, self.task_dataset_processing_params[task].seed)

            logger.info(f"Prepare single-task dataset for task '{task}' with {len(df)} data points.")

            logger.info("Filtering the molecules for Hydrogen")
            logger.info(f"Looking at column {df.columns[0]}")
            logger.info("Filtering done")
            # Extract smiles, labels, extras
            args = self.task_dataset_processing_params[task]
            smiles, labels, label_offsets, sample_idx, extras = self._extract_smiles_labels(
                df,
                task_level=args.task_level,
                smiles_col=args.smiles_col,
                label_cols=args.label_cols,
                idx_col=args.idx_col,
                weights_col=args.weights_col,
                weights_type=args.weights_type,
            )

            num_molecules = len(smiles)

            # Clear the reference to the DataFrame, so that Python can free up the memory.
            df = None

            # Store the relevant information for each task's dataset
            task_dataset_args[task] = {
                "smiles": smiles,
                "extras": extras,
            }
            if args.label_cols != 0:
                task_dataset_args[task]["labels"] = labels
                task_dataset_args[task]["label_offsets"] = label_offsets

            """We split the data up to create train, val and test datasets"""

            train_indices, val_indices, test_indices = self._get_split_indices(
                num_molecules,
                split_val=self.task_dataset_processing_params[task].split_val,
                split_test=self.task_dataset_processing_params[task].split_test,
                split_seed=self.task_dataset_processing_params[task].seed,
                splits_path=self.task_dataset_processing_params[task].splits_path,
                split_names=self.task_dataset_processing_params[task].split_names,
                # smiles and labels are already sub-sampled, so the split indices need to be
                # relative to the sample, not the original.
                # sample_idx=task_dataset_args[task]["sample_idx"],
            )
            self.task_train_indices[task] = train_indices
            self.task_val_indices[task] = val_indices
            self.task_test_indices[task] = test_indices

        logger.info("Done reading datasets")

        # The rest of the data preparation and caching is done in graphium_cpp.prepare_and_save_data
        normalizations = {
            task: self.task_dataset_processing_params[task].label_normalization
            for task in self.task_dataset_processing_params.keys()
        }
        (
            self.stage_data,
            all_stats,
            self.label_num_cols,
            self.label_dtypes,
        ) = graphium_cpp.prepare_and_save_data(
            self.task_names,
            task_dataset_args,
            normalizations,
            self.processed_graph_data_path,
            self.data_hash,
            self.task_train_indices,
            self.task_val_indices,
            self.task_test_indices,
            self.add_self_loop,
            self.explicit_H,
            self.preprocessing_n_jobs,
            self.merge_equivalent_mols,
        )

        for task, stats in all_stats.items():
            if len(stats) < 4:
                raise RuntimeError(f'Error loading cached stats for task "{task}"')

            self.task_norms[task].set_statistics(stats[0], stats[1], stats[2], stats[3])

        self._data_is_prepared = True

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
        label_num_cols = {}
        label_dtypes = {}
        if stage == "fit" or stage is None:
            if self.train_ds is None:
                self.train_ds = self._make_multitask_dataset("train")

            if self.val_ds is None and len(self.stage_data["val"]) >= self.num_edges_tensor_index():
                self.val_ds = self._make_multitask_dataset("val")

            logger.info(self.train_ds)
            label_num_cols.update(
                dict(zip(self.train_ds.task_names, self.train_ds.label_num_cols))
            )  # Make sure that all task label sizes are contained in here. Maybe do the update outside these if statements.
            label_dtypes.update(dict(zip(self.train_ds.task_names, self.train_ds.label_dtypes)))

            if self.val_ds is not None:
                logger.info(self.val_ds)
                label_num_cols.update(dict(zip(self.val_ds.task_names, self.val_ds.label_num_cols)))
                label_dtypes.update(dict(zip(self.val_ds.task_names, self.val_ds.label_dtypes)))

        if stage == "test" or stage is None:
            if self.test_ds is None and len(self.stage_data["test"]) >= self.num_edges_tensor_index():
                self.test_ds = self._make_multitask_dataset("test")

            if self.test_ds is not None:
                logger.info(self.test_ds)

                label_num_cols.update(dict(zip(self.test_ds.task_names, self.test_ds.label_num_cols)))
                label_dtypes.update(dict(zip(self.test_ds.task_names, self.test_ds.label_dtypes)))

        default_labels_num_cols_dict = self.collate_fn.keywords.get("labels_num_cols_dict", None)

        if default_labels_num_cols_dict is None:
            self.collate_fn.keywords["labels_num_cols_dict"] = label_num_cols

        default_labels_dtype_dict = self.collate_fn.keywords.get("labels_dtype_dict", None)

        if default_labels_dtype_dict is None:
            self.collate_fn.keywords["labels_dtype_dict"] = label_dtypes

    def _make_multitask_dataset(
        self,
        stage: Literal["train", "val", "test"],
    ) -> Datasets.MultitaskDataset:
        """
        Create a MultitaskDataset for the given stage using single task datasets
        The single task datasets must exist before this can be used

        Parameters:
            stage: Stage to create multitask dataset for
            processed_graph_data_path: path to save and load processed graph data from
        """

        allowed_stages = ["train", "val", "test"]
        assert stage in allowed_stages, f"Multitask dataset stage `{stage}` not in {allowed_stages}"

        if stage == "train":
            about = "training set"
        elif stage == "val":
            about = "validation set"
        elif stage == "test":
            about = "test set"
        else:
            raise ValueError(f"Unknown stage {stage}")

        processed_graph_data_path = self.processed_graph_data_path

        stage_data = self.stage_data[stage]
        data_offsets = None
        if self.data_offsets_tensor_index() < len(stage_data):
            data_offsets = stage_data[self.data_offsets_tensor_index()]

        multitask_dataset = Datasets.MultitaskDataset(
            about=about,
            data_path=self._path_to_load_from_file(stage) if processed_graph_data_path else None,
            featurize_smiles=self.smiles_transformer,
            task_names=self.task_names,
            label_num_cols=self.label_num_cols,
            label_dtypes=self.label_dtypes,
            mol_file_data_offsets=data_offsets,
            concat_smiles_tensor=stage_data[self.concat_smiles_tensor_index()],
            smiles_offsets_tensor=stage_data[self.smiles_offsets_tensor_index()],
            num_nodes_tensor=stage_data[self.num_nodes_tensor_index()],
            num_edges_tensor=stage_data[self.num_edges_tensor_index()],
        )  # type: ignore

        return multitask_dataset

    def _ready_to_load_all_from_file(self) -> bool:
        """
        Check if the data for all stages is ready to be loaded from files
        """

        paths = [self._path_to_load_from_file(stage) for stage in ["train", "val", "test"]]
        ready = all(self._data_ready_at_path(path) for path in paths)

        return ready

    def _path_to_load_from_file(self, stage: Literal["train", "val", "test"]) -> Optional[str]:
        """
        Get path from which to load the data from files
        """
        if self.processed_graph_data_path is None:
            return None
        return osp.join(self.processed_graph_data_path, f"{stage}_{self.data_hash}")

    def _data_ready_at_path(self, path: str) -> bool:
        """
        Check if data can be loaded from this path
        """
        can_load_from_file = osp.exists(path) and self.get_folder_size(path) > 0

        return can_load_from_file

    def get_folder_size(self, path):
        # check if the data items are actually saved into the folders
        return sum(os.path.getsize(osp.join(path, f)) for f in os.listdir(path))

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
        sampler = None
        # use sampler only when sampler_task_dict is set in the config and during training
        if DatasetSubSampler.check_sampling_required(self.sampler_task_dict) and stage in [
            RunningStage.TRAINING
        ]:
            sampler = DatasetSubSampler(
                dataset, self.sampler_task_dict, self.processed_graph_data_path, self.data_hash
            )
            # turn shuffle off when sampler is used as sampler option is mutually exclusive with shuffle
            kwargs["shuffle"] = False
        is_ipu = ("ipu_options" in kwargs.keys()) and (kwargs.get("ipu_options") is not None)
        if is_ipu:
            loader = IPUDataModuleModifier._dataloader(self, dataset=dataset, sampler=sampler, **kwargs)
        else:
            loader = BaseDataModule._dataloader(self, dataset=dataset, sampler=sampler, **kwargs)

        return loader

    def get_collate_fn(self, collate_fn):
        if collate_fn is None:
            # Some values become `inf` when changing data type. `mask_nan` deals with that
            collate_fn = partial(
                graphium_collate_fn,
                mask_nan=0,
                do_not_collate_keys=["smiles", "mol_ids"],
                batch_size_per_pack=self.batch_size_per_pack,
            )
            collate_fn.__name__ = graphium_collate_fn.__name__
        return collate_fn

    def _parse_label_cols(
        self,
        df: pd.DataFrame,
        df_path: Optional[Union[str, os.PathLike, List[Union[str, os.PathLike]]]],
        label_cols: Union[Type[None], str, List[str]],
        smiles_col: str,
    ) -> List[str]:
        r"""
        Parse the choice of label columns depending on the type of input.
        The input parameters `label_cols` and `smiles_col` are described in
        the `__init__` method.
        Parameters:
            df: The dataframe containing the labels.
            df_path: The path to the dataframe containing the labels. If list, the first file is used.
            label_cols: The columns to use as labels.
            smiles_col: The column to use as SMILES
        Returns:
            the parsed label columns
        """
        if df is None:
            files = self._glob(df_path)
            if len(files) == 0:
                raise FileNotFoundError(f"No such file or directory `{df_path}`")

            cols = BaseDataModule._get_table_columns(files[0])
            for file in files[1:]:
                _cols = BaseDataModule._get_table_columns(file)
                if set(cols) != set(_cols):
                    raise RuntimeError(
                        f"Multiple data files have different columns. \nColumn set 1: {cols}\nColumn set 2: {_cols}"
                    )
        else:
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
        graph = self.get_fake_graph()
        num_feats = graph.feat.shape[1]
        return num_feats

    @property
    def in_dims(self):
        """
        Return all input dimensions for the set of graphs.
        Including node/edge features, and
        raw positional encoding dimensions such eigval, eigvec, rwse and more
        """

        graph = self.get_fake_graph()

        # get list of all keys corresponding to positional encoding
        pe_dim_dict = {}
        g_keys = get_keys(graph)
        # ignore the normal keys for node feat and edge feat etc.
        for key in g_keys:
            prop = graph.get(key, None)
            if hasattr(prop, "shape"):
                pe_dim_dict[key] = prop.shape[-1]
        return pe_dim_dict

    @property
    def num_edge_feats(self):
        """Return the number of edge features in the first graph"""

        graph = self.get_fake_graph()
        empty = torch.Tensor([])
        num_feats = graph.get("edge_feat", empty).shape[-1]

        return num_feats

    def get_fake_graph(self):
        """
        Low memory footprint method to get the featurization of a fake graph
        without reading the dataset. Useful for getting the number of node/edge features.

        Returns:
            graph: A fake graph with the right featurization
        """

        smiles = "C1=CC=CC=C1"
        trans = deepcopy(self.smiles_transformer)
        trans.keywords.setdefault("on_error", "raise")
        trans.keywords.setdefault("mask_nan", 0.0)
        graph = trans(smiles)
        return graph

    ########################## Private methods ######################################

    @staticmethod
    def _extract_smiles_labels(
        df: pd.DataFrame,
        task_level: str,
        smiles_col: Optional[str] = None,
        label_cols: List[str] = [],
        idx_col: Optional[str] = None,
        mol_ids_col: Optional[str] = None,
        weights_col: Optional[str] = None,
        weights_type: Optional[str] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Union[Type[None], np.ndarray],
        Dict[str, Union[Type[None], np.ndarray]],
    ]:
        """
        For a given dataframe extract the SMILES and labels columns. Smiles is returned as a list
        of string while labels are returned as a 2D numpy array.

        Parameters:
            df: Pandas dataframe
            smiles_col: Name of the column containing the SMILES
            label_cols: List of column names containing the labels
            idx_col: Name of the column containing the index
            mol_ids_col: Name of the column containing the molecule ids
            weights_col: Name of the column containing the weights
            weights_type: Type of weights to use.
        Returns:
            smiles, labels, label_offsets, sample_idx, extras
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
            if task_level == "graph":
                labels, label_offsets = extract_labels(df, "graph", label_cols)
            elif task_level == "node":
                labels, label_offsets = extract_labels(df, "node", label_cols)
            elif task_level == "edge":
                labels, label_offsets = extract_labels(df, "edge", label_cols)
            elif task_level == "nodepair":
                labels, label_offsets = extract_labels(df, "nodepair", label_cols)
            else:
                raise ValueError(f"Unknown task level: {task_level}")
        else:
            labels = float("nan") + np.zeros([len(smiles), 0])
            label_offsets = None

        # Get the indices, used for sub-sampling and splitting the dataset
        if idx_col is not None:
            df = df.set_index(idx_col)
        sample_idx = df.index.values

        # Get the molecule ids
        mol_ids = None
        if mol_ids_col is not None:
            mol_ids = df[mol_ids_col].values

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

        extras = {"weights": weights, "mol_ids": mol_ids}
        return smiles, labels, label_offsets, sample_idx, extras

    @staticmethod
    def _get_split_indices(
        dataset_size: int,
        split_val: float,
        split_test: float,
        sample_idx: Optional[Iterable[int]] = None,
        split_seed: int = None,
        splits_path: Union[str, os.PathLike, Dict[str, Iterable[int]]] = None,
        split_names: Optional[List[str]] = ["train", "val", "test"],
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
            if split_test + split_val > 0:
                train_indices, val_test_indices = train_test_split(
                    sample_idx,
                    test_size=split_val + split_test,
                    random_state=split_seed,
                )
                sub_split_test = split_test / (split_test + split_val)
            else:
                train_indices = sample_idx
                val_test_indices = np.array([])
                sub_split_test = 0

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
            train, val, test = split_names
            if isinstance(splits_path, (Dict, pd.DataFrame)):
                # Split from a dataframe
                splits = splits_path
            else:
                # Split from an indices file
                file_type = BaseDataModule._get_data_file_type(splits_path)

                train, val, test = split_names

                if file_type == "pt":
                    splits = torch.load(splits_path)
                elif file_type in ["csv", "tsv"]:
                    with fsspec.open(str(splits_path)) as f:
                        splits = BaseDataModule._read_csv(splits_path)
                else:
                    raise ValueError(
                        f"file type `{file_type}` for `{splits_path}` not recognised, please use .pt, .csv or .tsv"
                    )
            train_indices = np.asarray(splits[train]).astype("int")
            train_indices = train_indices[~np.isnan(train_indices)].tolist()
            val_indices = np.asarray(splits[val]).astype("int")
            val_indices = val_indices[~np.isnan(val_indices)].tolist()
            test_indices = np.asarray(splits[test]).astype("int")
            test_indices = test_indices[~np.isnan(test_indices)].tolist()

        # Filter train, val and test indices
        _, train_idx, _ = np.intersect1d(sample_idx, train_indices, return_indices=True)
        train_indices = train_idx.tolist()
        _, val_idx, _ = np.intersect1d(sample_idx, val_indices, return_indices=True)
        val_indices = val_idx.tolist()
        _, test_idx, _ = np.intersect1d(sample_idx, test_indices, return_indices=True)
        test_indices = test_idx.tolist()

        return train_indices, val_indices, test_indices

    @staticmethod
    def _sub_sample_df(
        df: pd.DataFrame, sample_size: Union[int, float, None], seed: Optional[int] = None
    ) -> pd.DataFrame:
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
            df = df.sample(n=n, random_state=seed)
        elif isinstance(sample_size, float):
            df = df.sample(frac=sample_size, random_state=seed)
        elif sample_size is None:
            pass
        else:
            raise ValueError(
                f"Wrong value for `sample_size`: {sample_size}"
            )  # Maybe specify which task it was for?

        return df

    def get_data_hash(self):
        """
        Get a hash specific to a dataset.
        Useful to cache the pre-processed data.
        Don't include options only used at data loading time, such as
        most featurization options, but include options used during
        pre-processing, like merge_equivalent_mols.
        """
        args = {
            "add_self_loop": self.add_self_loop,
            "explicit_H": self.explicit_H,
            "merge_equivalent_mols": self.merge_equivalent_mols,
        }
        # pop epoch_sampling_fraction out when creating hash
        # so that the data cache does not need to be regenerated
        # when epoch_sampling_fraction has changed.
        for task_key, task_args in deepcopy(self.task_specific_args).items():
            if isinstance(task_args, DatasetProcessingParams):
                task_args = task_args.__dict__  # Convert the class to a dictionary

            # Keep only first 5 rows of a dataframe
            if "df" in task_args.keys():
                if task_args["df"] is not None:
                    task_args["df"] = task_args["df"].iloc[:5]

            # Remove the `epoch_sampling_fraction`
            task_args.pop("epoch_sampling_fraction", None)
            args[task_key] = task_args

        data_hash = get_md5_hash(args)
        return data_hash

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current DataModule, which is the combined size of all single-task datasets given.
        Returns:
            num_elements: Number of elements in the current DataModule
        """
        num_elements = 0
        for task, args in self.task_dataset_processing_params.items():
            if args.df is None:
                df = self._read_table(args.df_path, usecols=[args.smiles_col])
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
        obj_repr["batch_size_per_pack"] = self.batch_size_per_pack
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
        task_specific_args: Dict[str, Union[DatasetProcessingParams, Dict[str, Any]]],
        processed_graph_data_path: Optional[Union[str, os.PathLike]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        batch_size_per_pack: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        multiprocessing_context: Optional[str] = None,
        collate_fn: Optional[Callable] = None,
        preprocessing_n_jobs: int = 0,
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
            processed_graph_data_path: Path to the processed graph data. If None, the data will be
              downloaded from the OGB website.
            featurization: args to apply to the SMILES to Graph featurizer.
            batch_size_training: batch size for training and val dataset.
            batch_size_inference: batch size for test dataset.
            num_workers: Number of workers for the dataloader. Use -1 to use all available
                cores.
            pin_memory: Whether to pin on paginated CPU memory for the dataloader.
            collate_fn: A custom torch collate function. Default is to `graphium.data.graphium_collate_fn`
            sample_size:

                - `int`: The maximum number of elements to take from the dataset.
                - `float`: Value between 0 and 1 representing the fraction of the dataset to consider
                - `None`: all elements are considered.
            preprocessing_n_jobs: Number of threads to use during preprocessing.
                Use 0 to use all available cores, or -1 to use all but one core.

            dataloading_from: Deprecated. Behaviour now always matches previous "disk" option.
            featurization_n_jobs: Deprecated.
            featurization_progress: Deprecated.
            featurization_backend: Deprecated.
            prepare_dict_or_graph: Deprecated. Behaviour now always matches previous "pyg:graph" option.
        """

        new_task_specific_args = {}
        self.metadata = {}
        for task_name, task_args in task_specific_args.items():
            # Get OGB metadata
            this_metadata = self._get_dataset_metadata(task_args["dataset_name"])
            # Get dataset
            df, mol_ids_col, smiles_col, label_cols, splits_path = self._load_dataset(
                this_metadata, sample_size=task_args.get("sample_size", None)
            )
            new_task_specific_args[task_name] = {
                "df": df,
                "mol_ids_col": mol_ids_col,
                "smiles_col": smiles_col,
                "label_cols": label_cols,
                "splits_path": splits_path,
                "task_level": task_args["task_level"],
            }
            self.metadata[task_name] = this_metadata

        # Config for datamodule
        dm_args = {}
        dm_args["task_specific_args"] = new_task_specific_args
        dm_args["processed_graph_data_path"] = processed_graph_data_path
        dm_args["featurization"] = featurization
        dm_args["batch_size_training"] = batch_size_training
        dm_args["batch_size_inference"] = batch_size_inference
        dm_args["batch_size_per_pack"] = batch_size_per_pack
        dm_args["num_workers"] = num_workers
        dm_args["pin_memory"] = pin_memory
        dm_args["persistent_workers"] = persistent_workers
        dm_args["multiprocessing_context"] = multiprocessing_context
        dm_args["collate_fn"] = collate_fn
        dm_args["preprocessing_n_jobs"] = preprocessing_n_jobs

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
            mol_ids_col: Name of the column containing the molecule ids.
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
            mol_ids_col = df.columns[0]
            smiles_col = df.columns[-2]
            label_cols = df.columns[-1:].to_list()
        else:
            mol_ids_col = df.columns[-1]
            smiles_col = df.columns[-2]
            label_cols = df.columns[:-2].to_list()

        return df, mol_ids_col, smiles_col, label_cols, splits_path

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


class ADMETBenchmarkDataModule(MultitaskFromSmilesDataModule):
    """
    Wrapper to use the ADMET benchmark group from the TDC (Therapeutics Data Commons).

    !!! warning "Dependency"

        This class requires [PyTDC](https://pypi.org/project/PyTDC/) to be installed.

    !!! note "Citation"

        Huang, K., Fu, T., Gao, W., Zhao, Y., Roohani, Y., Leskovec, J., Coley, C., Xiao, C., Sun, J., & Zitnik, M. (2021).
        Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development.
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks.


    Parameters:
        tdc_benchmark_names: This can be any subset of the benchmark names that make up the ADMET benchmarking group.
            If `None`, uses the complete benchmarking group. For all full list of options, see
            [the TDC website](https://tdcommons.ai/benchmark/admet_group/overview/) or use:

           ```python
           import tdc.utils.retrieve_benchmark_names
           retrieve_benchmark_names("admet_group")
           ```
        tdc_train_val_seed: TDC recommends a default splitting method for the train-val split. This parameter
          is used to seed that splitting method.
    """

    def __init__(
        self,
        # TDC-specific
        tdc_benchmark_names: Optional[Union[str, List[str]]] = None,
        tdc_train_val_seed: int = 0,
        # Inherited arguments from superclass
        processed_graph_data_path: Optional[Union[str, Path]] = None,
        featurization: Optional[Union[Dict[str, Any], omegaconf.DictConfig]] = None,
        batch_size_training: int = 16,
        batch_size_inference: int = 16,
        batch_size_per_pack: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        multiprocessing_context: Optional[str] = None,
        collate_fn: Optional[Callable] = None,
        preprocessing_n_jobs: int = 0,
        **kwargs,
    ):
        try:
            from tdc.benchmark_group import admet_group
            from tdc.utils import retrieve_benchmark_names
        except ImportError as error:
            # To make sure we use the exact same train-test set and preprocessing as other benchmark entries,
            # we rely on the PyTDC library.
            raise RuntimeError(
                f"To use {self.__class__.__name__}, `PyTDC` needs to be installed. "
                f"Please install it with `pip install PyTDC`"
            ) from error

        # Pick a path to save the TDC data to
        tdc_cache_dir = fs.get_cache_dir("tdc")
        tdc_cache_dir = fs.join(tdc_cache_dir, "ADMET_Benchmark")
        fs.mkdir(tdc_cache_dir, exist_ok=True)

        # Create the benchmark group object
        # NOTE (cwognum): We redirect stderr and stdout to a file since TDC uses print statements,
        #  which quickly pollute the logs. Ideally, we would use `redirect_stderr(None)`, but that breaks TQDM.
        with tempfile.TemporaryFile("w") as f:
            with redirect_stderr(f):
                with redirect_stdout(f):
                    self.group = admet_group(path=tdc_cache_dir)

        # By default, use all available benchmarks in a benchmark group
        if tdc_benchmark_names is None:
            tdc_benchmark_names = retrieve_benchmark_names("admet_group")
        if isinstance(tdc_benchmark_names, str):
            tdc_benchmark_names = [tdc_benchmark_names]

        # Create the task-specific arguments
        logger.info(
            f"Preparing the TDC ADMET Benchmark Group splits for each of the {len(tdc_benchmark_names)} benchmarks."
        )

        task_specific_args = {
            t: self._get_task_specific_arguments(t, tdc_train_val_seed, tdc_cache_dir)
            for t in tdc_benchmark_names
        }

        super().__init__(
            task_specific_args=task_specific_args,
            featurization=featurization,
            processed_graph_data_path=processed_graph_data_path,
            batch_size_training=batch_size_training,
            batch_size_inference=batch_size_inference,
            batch_size_per_pack=batch_size_per_pack,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            collate_fn=collate_fn,
            preprocessing_n_jobs=preprocessing_n_jobs,
            **kwargs,
        )

    def _get_task_specific_arguments(self, name: str, seed: int, cache_dir: str) -> DatasetProcessingParams:
        """
        Loads the data and split from the TDC benchmark group object.

        For the train-test split, this is fixed.

        For the train-val split, this does not have to be fixed. Here we use the default splitting method
        from TDC to do the train-val split and allow the seed to be changed. This is likely to best match
        other entries in the benchmarking group.
        """

        benchmark = self.group.get(name)

        # Get the default train-val-test split

        # NOTE (cwognum): TDC prints by default to stderr, which pollutes the logs quite a bit.
        #  This context manager mutes these by temporarily writing to a file.
        #  Ideally, we would use `redirect_stderr(None)`, but that breaks TQDM.

        with tempfile.TemporaryFile("w") as f:
            with redirect_stderr(f):
                train, val = self.group.get_train_valid_split(seed, name)
        test = benchmark["test"]

        # Convert to the Graphium format
        n_val = len(val)
        n_test = len(test)
        n_train = len(train)
        max_len = max(n_train, n_val, n_test)
        total_len = n_train + n_val + n_test

        data = pd.concat([train, val, test], ignore_index=True)

        # NOTE (cwognum): We need to convert the labels to float, since we use NaNs down the line.
        #  If you uncomment this line, collating the labels will raise an overflow by converting a NaN to the int dtype.
        data["Y"] = data["Y"].astype(float)

        split = pd.DataFrame(
            {
                "train": list(range(n_train)),
                "val": list(range(n_train, n_train + n_val)) + [float("nan")] * (max_len - n_val),
                "test": list(range(n_train + n_val, total_len)) + [float("nan")] * (max_len - n_test),
            }
        )
        split_path = fs.join(cache_dir, f"{name}_split.csv")
        split.to_csv(split_path, index=False)

        return DatasetProcessingParams(
            df=data,
            idx_col=None,
            smiles_col="Drug",
            label_cols=["Y"],
            splits_path=split_path,
            split_names=["train", "val", "test"],
            task_level="graph",
        )
