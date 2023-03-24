import unittest as ut
import numpy as np
import torch
import pandas as pd

import goli
from goli.utils.fs import rm, exists, get_size
from goli.data import GraphOGBDataModule

TEMP_CACHE_DATA_PATH = "tests/temp_cache_0000"


class Test_DataModule(ut.TestCase):
    # TODO: Add this test once the OGB Datamodule is fixed
    def test_ogb_datamodule(self):
        # other datasets are too large to be tested
        dataset_names = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molfreesolv"]
        dataset_name = dataset_names[3]

        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["conformer_property_list"] = ["positions_3d"]
        featurization_args["edge_property_list"] = ["bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for datamodule
        task_specific_args = {}
        task_specific_args["task_1"] = {"dataset_name": dataset_name}
        dm_args = {}
        dm_args["cache_data_path"] = None
        dm_args["featurization"] = featurization_args
        dm_args["batch_size_training"] = 16
        dm_args["batch_size_inference"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "loky"

        ds = GraphOGBDataModule(task_specific_args, **dm_args)

        ds.prepare_data()

        # Check the keys in the dataset
        ds.setup(save_smiles_and_ids=False)
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        ds.setup(save_smiles_and_ids=True)
        assert set(ds.train_ds[0].keys()) == {"smiles", "mol_ids", "features", "labels"}

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert len(ds) == 642


        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["smiles"]) == 16
        assert len(batch["labels"]["task_1"]) == 16
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
            ) = goli.data.MultitaskFromSmilesDataModule._filter_none_molecules(
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
        featurization_args["conformer_property_list"] = ["positions_3d"]
        featurization_args["edge_property_list"] = ["bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for datamodule
        task_specific_args = {}
        task_specific_args["task_1"] = {"dataset_name": dataset_name}
        dm_args = {}
        dm_args["featurization"] = featurization_args
        dm_args["batch_size_training"] = 16
        dm_args["batch_size_inference"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "loky"

        # Delete the cache if already exist
        if exists(TEMP_CACHE_DATA_PATH):
            rm(TEMP_CACHE_DATA_PATH, recursive=True)

        # Prepare the data. It should create the cache there
        assert not exists(TEMP_CACHE_DATA_PATH)
        ds = GraphOGBDataModule(task_specific_args, cache_data_path=TEMP_CACHE_DATA_PATH, **dm_args)
        assert not ds.load_data_from_cache(verbose=False)
        ds.prepare_data()

        # Check the keys in the dataset
        ds.setup(save_smiles_and_ids=False)
        assert set(ds.train_ds[0].keys()) == {"features", "labels"}

        ds.setup(save_smiles_and_ids=True)
        assert set(ds.train_ds[0].keys()) == {"smiles", "mol_ids", "features", "labels"}


        # Make sure that the cache is created
        full_cache_path = ds.get_data_cache_fullname(compress=False)
        assert exists(full_cache_path)
        assert get_size(full_cache_path) > 10000

        # Check that the data is loaded correctly from cache
        assert ds.load_data_from_cache(verbose=False)

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert len(ds) == 642

        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["smiles"]) == 16
        assert len(batch["labels"]["task_1"]) == 16
        assert len(batch["mol_ids"]) == 16


if __name__ == "__main__":
    ut.main()

    # Delete the cache
    if exists(TEMP_CACHE_DATA_PATH):
        rm(TEMP_CACHE_DATA_PATH, recursive=True)
