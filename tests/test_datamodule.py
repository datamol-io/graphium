"""
--------------------------------------------------------------------------------
Copyright (c) 2023 Valence Labs, Recursion Pharmaceuticals and Graphcore Limited.

Use of this software is subject to the terms and conditions outlined in the LICENSE file.
Unauthorized modification, distribution, or use is prohibited. Provided 'as is' without
warranties of any kind.

Valence Labs, Recursion Pharmaceuticals and Graphcore Limited are not liable for any damages arising from its use.
Refer to the LICENSE file for the full terms and conditions.
--------------------------------------------------------------------------------
"""

import unittest as ut
import numpy as np
import torch
import pandas as pd
import tempfile

import graphium
from graphium.utils.fs import rm, exists, get_size
from graphium.data import GraphOGBDataModule, MultitaskFromSmilesDataModule

import graphium_cpp

TEMP_CACHE_DATA_PATH = "tests/temp_cache_0000"


class test_DataModule(ut.TestCase):
    def test_ogb_datamodule(self):
        # other datasets are too large to be tested
        dataset_names = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molfreesolv"]
        dataset_name = dataset_names[3]

        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        # featurization_args["conformer_property_list"] = ["positions_3d"]
        featurization_args["edge_property_list"] = ["bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for datamodule
        task_specific_args = {}
        task_specific_args["task_1"] = {"task_level": "graph", "dataset_name": dataset_name}
        dm_args = {}
        dm_args["featurization"] = featurization_args
        dm_args["batch_size_training"] = 16
        dm_args["batch_size_inference"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ds = GraphOGBDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, **dm_args)

        ds.prepare_data()

        # Check the keys in the dataset
        ds.setup()
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Reset the datamodule
        ds = GraphOGBDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, **dm_args)

        ds.prepare_data()

        # Check the keys in the dataset
        ds.setup()
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert (
            len(ds) == 642 or len(ds) == 644
        )  # Accounting for differences in csv file reads across Linux & OSX

        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["labels"]["graph_task_1"]) == 16

    def test_caching(self):
        # other datasets are too large to be tested
        dataset_name = "ogbg-molfreesolv"

        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for datamodule
        task_specific_args = {}
        task_specific_args["task_1"] = {"task_level": "graph", "dataset_name": dataset_name}
        dm_args = {}
        dm_args["featurization"] = featurization_args
        dm_args["batch_size_training"] = 16
        dm_args["batch_size_inference"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Prepare the data. It should create the cache there
        assert not exists(TEMP_CACHE_DATA_PATH)
        ds = GraphOGBDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, **dm_args)
        # assert not ds.load_data_from_cache(verbose=False)
        ds.prepare_data()

        # Check the keys in the dataset
        ds.setup()
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        # ds_batch = next(iter(ds.train_dataloader()))
        train_loader = ds.get_dataloader(ds.train_ds, shuffle=False, stage="train")
        batch = next(iter(train_loader))

        # Test loading cached data
        assert exists(TEMP_CACHE_DATA_PATH)

        cached_ds_from_disk = GraphOGBDataModule(
            task_specific_args,
            processed_graph_data_path=TEMP_CACHE_DATA_PATH,
            **dm_args,
        )
        cached_ds_from_disk.prepare_data()
        cached_ds_from_disk.setup()
        cached_train_loader_from_disk = cached_ds_from_disk.get_dataloader(
            cached_ds_from_disk.train_ds, shuffle=False, stage="train"
        )
        batch_from_disk = next(iter(cached_train_loader_from_disk))

        # Features are the same
        np.testing.assert_array_almost_equal(
            batch["features"].edge_index, batch_from_disk["features"].edge_index
        )

        assert batch["features"].num_nodes == batch_from_disk["features"].num_nodes

        np.testing.assert_array_almost_equal(
            batch["features"].edge_weight, batch_from_disk["features"].edge_weight
        )

        np.testing.assert_array_almost_equal(batch["features"].feat, batch_from_disk["features"].feat)

        np.testing.assert_array_almost_equal(
            batch["features"].edge_feat, batch_from_disk["features"].edge_feat
        )

        np.testing.assert_array_almost_equal(batch["features"].batch, batch_from_disk["features"].batch)

        np.testing.assert_array_almost_equal(batch["features"].ptr, batch_from_disk["features"].ptr)

        # Labels are the same
        np.testing.assert_array_almost_equal(
            batch["labels"].graph_task_1, batch_from_disk["labels"].graph_task_1
        )

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Reset the datamodule
        ds = GraphOGBDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, **dm_args)

        ds.prepare_data()

        ds.setup()
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert (
            len(ds) == 642 or len(ds) == 644
        )  # Accounting for differences in csv file reads across Linux & OSX

        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["labels"]["graph_task_1"]) == 16

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

    def test_datamodule_with_none_molecules(self):
        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["bond-type-onehot"]

        # Config for datamodule
        bad_csv = "tests/data/micro_ZINC_corrupt.csv"
        task_specific_args = {}
        task_kwargs = {"df_path": bad_csv, "split_val": 0.0, "split_test": 0.0}
        task_specific_args["task_1"] = {
            "task_level": "graph",
            "label_cols": "SA",
            "smiles_col": "SMILES1",
            **task_kwargs,
        }
        task_specific_args["task_2"] = {
            "task_level": "graph",
            "label_cols": "logp",
            "smiles_col": "SMILES2",
            **task_kwargs,
        }
        task_specific_args["task_3"] = {
            "task_level": "graph",
            "label_cols": "score",
            "smiles_col": "SMILES3",
            **task_kwargs,
        }

        # Read the corrupted dataset and get stats
        df = pd.read_csv(bad_csv)
        bad_smiles = (df["SMILES1"] == "XXX") & (df["SMILES2"] == "XXX") & (df["SMILES3"] == "XXX")
        num_bad_smiles = sum(bad_smiles)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Test the datamodule
        datamodule = MultitaskFromSmilesDataModule(
            task_specific_args=task_specific_args,
            processed_graph_data_path=TEMP_CACHE_DATA_PATH,
            featurization_args=featurization_args,
        )
        datamodule.prepare_data()
        datamodule.setup()

        # Check that the number of molecules is correct
        smiles = df["SMILES1"].tolist() + df["SMILES2"].tolist() + df["SMILES3"].tolist()
        num_unique_smiles = len(set(smiles)) - 1  # -1 because of the XXX
        # self.assertEqual(len(datamodule.train_ds), num_unique_smiles - num_bad_smiles)

        # Change the index of the dataframe
        index_smiles = []
        for ii in range(len(df)):
            if df["SMILES1"][ii] != "XXX":
                smiles = df["SMILES1"][ii]
            elif df["SMILES2"][ii] != "XXX":
                smiles = df["SMILES2"][ii]
            elif df["SMILES3"][ii] != "XXX":
                smiles = df["SMILES3"][ii]
            else:
                smiles = "XXX"
            index_smiles.append(smiles)
        df["idx_smiles"] = index_smiles
        df = df.set_index("idx_smiles")

        # Convert the smilies from the train_ds to a list, and check the content
        train_smiles = [
            graphium_cpp.extract_string(
                datamodule.train_ds.smiles_tensor, datamodule.train_ds.smiles_offsets_tensor, idx
            )
            for idx in range(len(datamodule.train_ds))
        ]

        # Check that the set of smiles are the same
        train_smiles_flat = list(set(train_smiles))
        train_smiles_flat.sort()
        index_smiles_filt = list(set([smiles for smiles in index_smiles if smiles != "XXX"]))
        index_smiles_filt.sort()
        self.assertListEqual(train_smiles_flat, index_smiles_filt)

        # Check that the smiles is correct for each datapoint in the dataset
        for smiles in train_smiles:
            assert isinstance(smiles, str)
            true_smiles = df.loc[smiles][["SMILES1", "SMILES2", "SMILES3"]]
            self.assertEqual(
                smiles, true_smiles[true_smiles != "XXX"].values[0]
            )  # Check that the smiles is correct

        # Convert the labels from the train_ds to a dataframe
        train_labels = [datamodule.train_ds[idx]["labels"] for idx in range(len(datamodule.train_ds))]
        train_labels = [{k: v[0].item() for k, v in label} for label in train_labels]
        train_labels_df = pd.DataFrame(train_labels)
        train_labels_df = train_labels_df.rename(
            columns={"graph_task_1": "graph_SA", "graph_task_2": "graph_logp", "graph_task_3": "graph_score"}
        )
        train_labels_df["smiles"] = train_smiles
        train_labels_df = train_labels_df.set_index("smiles")
        train_labels_df = train_labels_df.sort_index()

        # Check that the labels are correct
        df2 = df.reset_index()[~bad_smiles].set_index("idx_smiles").sort_index()
        labels = train_labels_df[["graph_SA", "graph_logp", "graph_score"]].values
        nans = np.isnan(labels)
        true_nans = df2[["SMILES1", "SMILES2", "SMILES3"]].values == "XXX"
        true_labels = df2[["SA", "logp", "score"]].values
        true_labels[true_nans] = np.nan
        np.testing.assert_array_equal(nans, true_nans)  # Check that the nans are correct
        np.testing.assert_array_almost_equal(
            labels, true_labels, decimal=5
        )  # Check that the label values are correct

    def test_datamodule_multiple_data_files(self):
        # Test single CSV files
        csv_file = "tests/data/micro_ZINC_shard_1.csv"
        task_kwargs = {"df_path": csv_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 10)

        # Test multi CSV files
        csv_file = "tests/data/micro_ZINC_shard_*.csv"
        task_kwargs = {"df_path": csv_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 20)

        # Test single Parquet files
        parquet_file = "tests/data/micro_ZINC_shard_1.parquet"
        task_kwargs = {"df_path": parquet_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 10)

        # Test multi Parquet files
        parquet_file = "tests/data/micro_ZINC_shard_*.parquet"
        task_kwargs = {"df_path": parquet_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 20)

    def test_splits_file(self, tmp_path):
        # Test single CSV files
        csv_file = "tests/data/micro_ZINC_shard_1.csv"
        df = pd.read_csv(csv_file)

        # Split the CSV file with 80/10/10
        train = 0.8
        val = 0.1
        indices = np.arange(len(df))
        split_train = indices[: int(len(df) * train)]
        split_val = indices[int(len(df) * train) : int(len(df) * (train + val))]
        split_test = indices[int(len(df) * (train + val)) :]

        splits = {"train": split_train, "val": split_val, "test": split_test}

        # Test the splitting using `splits` directly as `splits_path`
        task_kwargs = {
            "df_path": csv_file,
            "splits_path": splits,
            "split_val": 0.0,
            "split_test": 0.0,
        }
        task_specific_args = {
            "task": {
                "task_level": "graph",
                "label_cols": ["score"],
                "smiles_col": "SMILES",
                **task_kwargs,
            }
        }

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        ds = MultitaskFromSmilesDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), len(split_train))
        self.assertEqual(len(ds.val_ds), len(split_val))
        self.assertEqual(len(ds.test_ds), len(split_test))

        # Create a TemporaryFile to save the splits, and test the datamodule
        with tempfile.NamedTemporaryFile(suffix=".pt", dir=tmp_path) as temp:
            # Save the splits
            torch.save(splits, temp)

            # Test the datamodule
            task_kwargs = {
                "df_path": csv_file,
                "splits_path": temp.name,
                "split_val": 0.0,
                "split_test": 0.0,
            }
            task_specific_args = {
                "task": {
                    "task_level": "graph",
                    "label_cols": ["score"],
                    "smiles_col": "SMILES",
                    **task_kwargs,
                }
            }

            # Delete the cache if already exist
            if exists(TEMP_CACHE_DATA_PATH):
                rm(TEMP_CACHE_DATA_PATH, recursive=True)

            ds2 = MultitaskFromSmilesDataModule(
                task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH
            )
            ds2.prepare_data()
            ds2.setup()

            self.assertEqual(len(ds2.train_ds), len(split_train))
            self.assertEqual(len(ds2.val_ds), len(split_val))
            self.assertEqual(len(ds2.test_ds), len(split_test))

            # Check that the splits are the same
            self.assertEqual(len(ds.train_ds.smiles_offsets_tensor), len(split_train) + 1)
            np.testing.assert_array_equal(ds.train_ds.smiles_tensor, ds2.train_ds.smiles_tensor)
            np.testing.assert_array_equal(ds.val_ds.smiles_tensor, ds2.val_ds.smiles_tensor)
            np.testing.assert_array_equal(ds.test_ds.smiles_tensor, ds2.test_ds.smiles_tensor)
            np.testing.assert_array_equal(
                ds.train_ds.smiles_offsets_tensor, ds2.train_ds.smiles_offsets_tensor
            )
            np.testing.assert_array_equal(ds.val_ds.smiles_offsets_tensor, ds2.val_ds.smiles_offsets_tensor)
            np.testing.assert_array_equal(ds.test_ds.smiles_offsets_tensor, ds2.test_ds.smiles_offsets_tensor)


if __name__ == "__main__":
    ut.main()

    # Delete the cache
    if exists(TEMP_CACHE_DATA_PATH):
        rm(TEMP_CACHE_DATA_PATH, recursive=True)
