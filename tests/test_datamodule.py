import pathlib
import tempfile

import unittest as ut
import numpy as np
import torch
import pandas as pd

import goli


class Test_DataModule(ut.TestCase):
    def test_dglfromsmiles_dm(self):

        df = goli.data.load_tiny_zinc()
        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["in-ring", "bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for datamodule
        dm_args = {}
        dm_args["df"] = df
        dm_args["cache_data_path"] = None
        dm_args["featurization"] = featurization_args
        dm_args["smiles_col"] = "SMILES"
        dm_args["label_cols"] = ["SA"]
        dm_args["split_val"] = 0.2
        dm_args["split_test"] = 0.2
        dm_args["split_seed"] = 19
        dm_args["batch_size_train_val"] = 16
        dm_args["batch_size_test"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "multiprocessing"

        dm = goli.data.DGLFromSmilesDataModule(**dm_args)

        assert dm.num_node_feats == 50
        assert dm.num_edge_feats == 6

        dm.prepare_data()
        dm.setup()

        assert len(dm) == 100
        assert len(dm.train_ds) == 60  # type: ignore
        assert len(dm.val_ds) == 20  # type: ignore
        assert len(dm.test_ds) == 20  # type: ignore
        assert dm.num_node_feats == 50
        assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)

            assert set(batch.keys()) == {"labels", "features", "smiles"}
            assert batch["labels"].shape == (16, 1)

    def test_dglfromsmiles_from_config(self):

        config = goli.load_config(name="zinc_default_fulldgl")
        df = goli.data.load_tiny_zinc()

        dm_args = dict(config.data.args)
        dm_args["df"] = df

        dm = goli.data.DGLFromSmilesDataModule(**dm_args)

        assert dm.num_node_feats == 50
        assert dm.num_edge_feats == 6

        dm.prepare_data()
        dm.setup()

        assert len(dm.train_ds) == 60  # type: ignore
        assert len(dm.val_ds) == 20  # type: ignore
        assert len(dm.test_ds) == 20  # type: ignore
        assert dm.num_node_feats == 50
        assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)

            assert set(batch.keys()) == {"labels", "features", "smiles"}
            assert batch["labels"].shape == (16, 1)

    def test_ogb_datamodule(self):

        # other datasets are too large to be tested
        dataset_names = ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molfreesolv"]
        dataset_name = dataset_names[3]

        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for datamodule
        dm_args = {}
        dm_args["dataset_name"] = dataset_name
        dm_args["cache_data_path"] = None
        dm_args["featurization"] = featurization_args
        dm_args["batch_size_train_val"] = 16
        dm_args["batch_size_test"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "loky"

        ds = goli.data.DGLOGBDataModule(**dm_args)

        ds.prepare_data()
        ds.setup()

        # test module
        assert ds.num_edge_feats == 5
        assert ds.num_node_feats == 50
        assert len(ds) == 642
        assert ds.dataset_name == "ogbg-molfreesolv"

        # test dataset
        assert set(ds.train_ds[0].keys()) == {"smiles", "indices", "features", "labels"}

        # test batch loader
        batch = next(iter(ds.train_dataloader()))
        assert len(batch["smiles"]) == 16
        assert len(batch["labels"]) == 16
        assert len(batch["indices"]) == 16

    def test_datamodule_cache_invalidation(self):

        df = goli.data.load_tiny_zinc()

        cache_data_path = pathlib.Path(tempfile.mkdtemp()) / "cache.pkl"

        # 1. Build a module with specific feature arguments

        featurization_args = {}
        featurization_args["atom_property_list_float"] = ["mass", "electronegativity", "in-ring"]
        featurization_args["edge_property_list"] = ["bond-type-onehot", "stereo", "in-ring"]

        dm_args = {}
        dm_args["df"] = df
        dm_args["cache_data_path"] = cache_data_path
        dm_args["featurization"] = featurization_args
        datam = goli.data.DGLFromSmilesDataModule(**dm_args)
        datam.prepare_data()
        datam.setup()

        assert datam.num_node_feats == 3
        assert datam.num_edge_feats == 13

        # 2. Reload with the same arguments should not trigger a new preparation and give
        # the same feature's dimensions.

        datam = goli.data.DGLFromSmilesDataModule(**dm_args)
        datam.prepare_data()
        datam.setup()

        assert datam.num_node_feats == 3
        assert datam.num_edge_feats == 13

        # 3. Reloading from the same cache file should trigger a new data preparation and
        # so different feature's dimensions.

        featurization_args = {}
        featurization_args["edge_property_list"] = ["stereo", "in-ring"]
        featurization_args["atom_property_list_float"] = ["mass", "electronegativity"]

        dm_args = {}
        dm_args["df"] = df
        dm_args["cache_data_path"] = cache_data_path
        dm_args["featurization"] = featurization_args
        datam = goli.data.DGLFromSmilesDataModule(**dm_args)
        datam.prepare_data()
        datam.setup()

        assert datam.num_node_feats == 2
        assert datam.num_edge_feats == 8

    def test_none_filtering(self):
        # Create the objects to filter
        list_of_num = [ii for ii in range(100)]
        list_of_str = [str(ii) for ii in list_of_num]
        tuple_of_num = tuple(list_of_num)
        array_of_num = np.stack([list_of_num, list_of_num, list_of_num], axis=1)
        array_of_str = np.stack([list_of_str, list_of_str, list_of_str], axis=1)
        tensor_of_num = torch.as_tensor(array_of_num)
        dic = {"str": list_of_str, "num": list_of_num}
        df = pd.DataFrame(dic)
        df_shuffled = df.sample(frac=1)

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
            ) = goli.data.DGLFromSmilesDataModule._filter_none_molecules(
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
            )

            df_shuffled_2 = df_shuffled_2.sort_values(by="num")

            # Assert the filtering is done correctly
            self.assertListEqual(list_of_num_2, filtered_num, msg=msg)
            self.assertListEqual(list_of_str_2, filtered_str, msg=msg)
            self.assertListEqual(list(tuple_of_num_2), filtered_num, msg=msg)
            for jj in range(array_of_num.shape[1]):
                self.assertListEqual(array_of_num_2[:, jj].tolist(), filtered_num, msg=msg)
                self.assertListEqual(array_of_str_2[:, jj].tolist(), filtered_str, msg=msg)
                self.assertListEqual(tensor_of_num_2[:, jj].tolist(), filtered_num, msg=msg)
            self.assertListEqual(dic_2["num"], filtered_num, msg=msg)
            self.assertListEqual(dic_2["str"], filtered_str, msg=msg)
            self.assertListEqual(df_2["num"].tolist(), filtered_num, msg=msg)
            self.assertListEqual(df_2["str"].tolist(), filtered_str, msg=msg)

            # When the dataframe is shuffled, the lists are different because the filtering
            # is done on the row indexes, not the dataframe indexes.
            if (len(idx_none) > 0) and (len(idx_none) < len(df_shuffled)):
                self.assertTrue(df_shuffled_2["num"].tolist() != filtered_num, msg=msg)
                self.assertTrue(df_shuffled_2["str"].tolist() != filtered_str, msg=msg)


if __name__ == "__main__":
    ut.main()
