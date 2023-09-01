import unittest as ut
import numpy as np
import torch
import pandas as pd
import datamol as dm

import graphium
from graphium.utils.fs import rm, exists, get_size
from graphium.data import GraphOGBDataModule, MultitaskFromSmilesDataModule

TEMP_CACHE_DATA_PATH = "tests/temp_cache_0000"


class Test_DataModule(ut.TestCase):
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
        dm_args["processed_graph_data_path"] = None
        dm_args["featurization"] = featurization_args
        dm_args["batch_size_training"] = 16
        dm_args["batch_size_inference"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["featurization_n_jobs"] = 0
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "loky"
        dm_args["featurization_batch_size"] = 50

        ds = GraphOGBDataModule(task_specific_args, **dm_args)

        ds.prepare_data(save_smiles_and_ids=False)

        # Check the keys in the dataset
        ds.setup(save_smiles_and_ids=False)
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Reset the datamodule
        ds = GraphOGBDataModule(task_specific_args, **dm_args)

        ds.prepare_data(save_smiles_and_ids=True)

        # Check the keys in the dataset
        ds.setup(save_smiles_and_ids=True)
        assert set(ds.train_ds[0].keys()) == {"smiles", "mol_ids", "features", "labels"}

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert len(ds) == 642

        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["smiles"]) == 16
        assert len(batch["labels"]["graph_task_1"]) == 16
        assert len(batch["mol_ids"]) == 16

    def test_none_filtering(self):
        # Create the objects to filter
        list_of_num = [ii for ii in range(100)]
        list_of_str = [str(ii) for ii in list_of_num]
        tuple_of_num = tuple(list_of_num)
        array_of_num = np.asarray(list_of_num)
        array_of_str = np.asarray(list_of_str)
        tensor_of_num = torch.as_tensor(array_of_num)
        arrays_of_num = np.stack([list_of_num, list_of_num, list_of_num], axis=1)
        arrays_of_str = np.stack([list_of_str, list_of_str, list_of_str], axis=1)
        tensors_of_num = torch.as_tensor(arrays_of_num)
        dic = {"str": list_of_str, "num": list_of_num}
        df = pd.DataFrame(dic)
        df_shuffled = df.sample(frac=1)
        series_num = df["num"]
        series_num_shuffled = df_shuffled["num"]

        # Create different indexes to use for filtering
        all_idx_none = [[3, 17, 88], [22, 33, 44, 55, 66, 77, 88], [], np.arange(len(list_of_num))]

        # Loop all the indexes and filter the objects.
        for ii, idx_none in enumerate(all_idx_none):
            msg = f"Failed for ii={ii}"

            # Create the true filtered sequences
            filtered_num = [ii for ii in range(100) if ii not in idx_none]
            filtered_str = [str(ii) for ii in filtered_num]
            assert len(filtered_num) == len(list_of_num) - len(idx_none)
            assert len(filtered_str) == len(list_of_str) - len(idx_none)

            # Filter the sequences from the Datamodule function
            (
                list_of_num_2,
                list_of_str_2,
                tuple_of_num_2,
                array_of_num_2,
                array_of_str_2,
                tensor_of_num_2,
                df_2,
                df_shuffled_2,
                dic_2,
                arrays_of_num_2,
                arrays_of_str_2,
                tensors_of_num_2,
                series_num_2,
                series_num_shuffled_2,
            ) = graphium.data.MultitaskFromSmilesDataModule._filter_none_molecules(
                idx_none,
                list_of_num,
                list_of_str,
                tuple_of_num,
                array_of_num,
                array_of_str,
                tensor_of_num,
                df,
                df_shuffled,
                dic,
                arrays_of_num,
                arrays_of_str,
                tensors_of_num,
                series_num,
                series_num_shuffled,
            )

            df_shuffled_2 = df_shuffled_2.sort_values(by="num", axis=0)
            series_num_shuffled_2 = series_num_shuffled_2.sort_values(axis=0)

            # Assert the filtering is done correctly
            self.assertListEqual(list_of_num_2, filtered_num, msg=msg)
            self.assertListEqual(list_of_str_2, filtered_str, msg=msg)
            self.assertListEqual(list(tuple_of_num_2), filtered_num, msg=msg)
            self.assertListEqual(array_of_num_2.tolist(), filtered_num, msg=msg)
            self.assertListEqual(array_of_str_2.tolist(), filtered_str, msg=msg)
            self.assertListEqual(tensor_of_num_2.tolist(), filtered_num, msg=msg)
            for jj in range(arrays_of_num.shape[1]):
                self.assertListEqual(arrays_of_num_2[:, jj].tolist(), filtered_num, msg=msg)
                self.assertListEqual(arrays_of_str_2[:, jj].tolist(), filtered_str, msg=msg)
                self.assertListEqual(tensors_of_num_2[:, jj].tolist(), filtered_num, msg=msg)
            self.assertListEqual(dic_2["num"], filtered_num, msg=msg)
            self.assertListEqual(dic_2["str"], filtered_str, msg=msg)
            self.assertListEqual(df_2["num"].tolist(), filtered_num, msg=msg)
            self.assertListEqual(df_2["str"].tolist(), filtered_str, msg=msg)
            self.assertListEqual(series_num_2.tolist(), filtered_num, msg=msg)

            # When the dataframe is shuffled, the lists are different because the filtering
            # is done on the row indexes, not the dataframe indexes.
            bool_to_check = (len(idx_none) == 0) or (len(idx_none) == len(df_shuffled))
            self.assertIs(df_shuffled_2["num"].tolist() == filtered_num, bool_to_check, msg=msg)
            self.assertIs(df_shuffled_2["str"].tolist() == filtered_str, bool_to_check, msg=msg)
            self.assertIs(series_num_shuffled_2.tolist() == filtered_num, bool_to_check, msg=msg)

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
        dm_args["featurization_n_jobs"] = 0
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "loky"
        dm_args["featurization_batch_size"] = 50

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Prepare the data. It should create the cache there
        assert not exists(TEMP_CACHE_DATA_PATH)
        ds = GraphOGBDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, **dm_args)
        # assert not ds.load_data_from_cache(verbose=False)
        ds.prepare_data(save_smiles_and_ids=False)

        # Check the keys in the dataset
        ds.setup(save_smiles_and_ids=False)
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        # ds_batch = next(iter(ds.train_dataloader()))
        train_loader = ds.get_dataloader(ds.train_ds, shuffle=False, stage="train")
        batch = next(iter(train_loader))

        # Test loading cached data
        assert exists(TEMP_CACHE_DATA_PATH)

        cached_ds_from_ram = GraphOGBDataModule(
            task_specific_args,
            processed_graph_data_path=TEMP_CACHE_DATA_PATH,
            dataloading_from="ram",
            **dm_args,
        )
        cached_ds_from_ram.prepare_data()
        cached_ds_from_ram.setup()
        cached_train_loader_from_ram = cached_ds_from_ram.get_dataloader(
            cached_ds_from_ram.train_ds, shuffle=False, stage="train"
        )
        batch_from_ram = next(iter(cached_train_loader_from_ram))

        cached_ds_from_disk = GraphOGBDataModule(
            task_specific_args,
            processed_graph_data_path=TEMP_CACHE_DATA_PATH,
            dataloading_from="disk",
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
            batch["features"].edge_index, batch_from_ram["features"].edge_index
        )
        np.testing.assert_array_almost_equal(
            batch["features"].edge_index, batch_from_disk["features"].edge_index
        )

        assert batch["features"].num_nodes == batch_from_ram["features"].num_nodes
        assert batch["features"].num_nodes == batch_from_disk["features"].num_nodes

        np.testing.assert_array_almost_equal(
            batch["features"].edge_weight, batch_from_ram["features"].edge_weight
        )
        np.testing.assert_array_almost_equal(
            batch["features"].edge_weight, batch_from_disk["features"].edge_weight
        )

        np.testing.assert_array_almost_equal(batch["features"].feat, batch_from_ram["features"].feat)
        np.testing.assert_array_almost_equal(batch["features"].feat, batch_from_disk["features"].feat)

        np.testing.assert_array_almost_equal(
            batch["features"].edge_feat, batch_from_ram["features"].edge_feat
        )
        np.testing.assert_array_almost_equal(
            batch["features"].edge_feat, batch_from_disk["features"].edge_feat
        )

        np.testing.assert_array_almost_equal(batch["features"].batch, batch_from_ram["features"].batch)
        np.testing.assert_array_almost_equal(batch["features"].batch, batch_from_disk["features"].batch)

        np.testing.assert_array_almost_equal(batch["features"].ptr, batch_from_ram["features"].ptr)
        np.testing.assert_array_almost_equal(batch["features"].ptr, batch_from_disk["features"].ptr)

        # Labels are the same
        np.testing.assert_array_almost_equal(
            batch["labels"].graph_task_1, batch_from_ram["labels"].graph_task_1
        )
        np.testing.assert_array_almost_equal(
            batch["labels"].graph_task_1, batch_from_disk["labels"].graph_task_1
        )

        np.testing.assert_array_almost_equal(batch["labels"].x, batch_from_ram["labels"].x)
        np.testing.assert_array_almost_equal(batch["labels"].x, batch_from_disk["labels"].x)

        np.testing.assert_array_almost_equal(batch["labels"].edge_index, batch_from_ram["labels"].edge_index)
        np.testing.assert_array_almost_equal(batch["labels"].edge_index, batch_from_disk["labels"].edge_index)

        np.testing.assert_array_almost_equal(batch["labels"].batch, batch_from_ram["labels"].batch)
        np.testing.assert_array_almost_equal(batch["labels"].batch, batch_from_disk["labels"].batch)

        np.testing.assert_array_almost_equal(batch["labels"].ptr, batch_from_ram["labels"].ptr)
        np.testing.assert_array_almost_equal(batch["labels"].ptr, batch_from_disk["labels"].ptr)

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Reset the datamodule
        ds = GraphOGBDataModule(task_specific_args, processed_graph_data_path=TEMP_CACHE_DATA_PATH, **dm_args)

        ds.prepare_data(save_smiles_and_ids=True)

        ds.setup(save_smiles_and_ids=True)
        assert set(ds.train_ds[0].keys()) == {"smiles", "mol_ids", "features", "labels"}

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert len(ds) == 642

        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["smiles"]) == 16
        assert len(batch["labels"]["graph_task_1"]) == 16
        assert len(batch["mol_ids"]) == 16

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

        # Test the datamodule
        datamodule = MultitaskFromSmilesDataModule(
            task_specific_args=task_specific_args,
            featurization_args=featurization_args,
            featurization_n_jobs=0,
            featurization_batch_size=1,
        )
        datamodule.prepare_data()
        datamodule.setup(save_smiles_and_ids=True)

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
        train_smiles = [d["smiles"] for d in datamodule.train_ds]

        # Check that the set of smiles are the same
        train_smiles_flat = list(set([item for sublist in train_smiles for item in sublist]))
        train_smiles_flat.sort()
        index_smiles_filt = list(set([smiles for smiles in index_smiles if smiles != "XXX"]))
        index_smiles_filt.sort()
        self.assertListEqual(train_smiles_flat, index_smiles_filt)

        # Check that the smiles are correct for each datapoint in the dataset
        for smiles in train_smiles:
            self.assertEqual(len(set(smiles)), 1)  # Check that all smiles are the same
            this_smiles = smiles[0]
            true_smiles = df.loc[this_smiles][["SMILES1", "SMILES2", "SMILES3"]]
            num_true_smiles = sum(true_smiles != "XXX")
            self.assertEqual(len(smiles), num_true_smiles)  # Check that the number of smiles is correct
            self.assertEqual(
                this_smiles, true_smiles[true_smiles != "XXX"].values[0]
            )  # Check that the smiles are correct

        # Convert the labels from the train_ds to a dataframe
        train_labels = [{task: val[0] for task, val in d["labels"].items()} for d in datamodule.train_ds]
        train_labels_df = pd.DataFrame(train_labels)
        train_labels_df = train_labels_df.rename(
            columns={"graph_task_1": "graph_SA", "graph_task_2": "graph_logp", "graph_task_3": "graph_score"}
        )
        train_labels_df["smiles"] = [s[0] for s in datamodule.train_ds.smiles]
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

        ds = MultitaskFromSmilesDataModule(task_specific_args, featurization_n_jobs=0)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 10)

        # Test multi CSV files
        csv_file = "tests/data/micro_ZINC_shard_*.csv"
        task_kwargs = {"df_path": csv_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        ds = MultitaskFromSmilesDataModule(task_specific_args, featurization_n_jobs=0)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 20)

        # Test single Parquet files
        parquet_file = "tests/data/micro_ZINC_shard_1.parquet"
        task_kwargs = {"df_path": parquet_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        ds = MultitaskFromSmilesDataModule(task_specific_args, featurization_n_jobs=0)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 10)

        # Test multi Parquet files
        parquet_file = "tests/data/micro_ZINC_shard_*.parquet"
        task_kwargs = {"df_path": parquet_file, "split_val": 0.0, "split_test": 0.0}
        task_specific_args = {
            "task": {"task_level": "graph", "label_cols": ["score"], "smiles_col": "SMILES", **task_kwargs}
        }

        ds = MultitaskFromSmilesDataModule(task_specific_args, featurization_n_jobs=0)
        ds.prepare_data()
        ds.setup()

        self.assertEqual(len(ds.train_ds), 20)


if __name__ == "__main__":
    ut.main()

    # Delete the cache
    if exists(TEMP_CACHE_DATA_PATH):
        rm(TEMP_CACHE_DATA_PATH, recursive=True)
