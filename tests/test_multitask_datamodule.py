import shutil
import tempfile
import unittest as ut

import pytest
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import graphium


class Test_Multitask_DataModule(ut.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.tmp_test_dir)

    def test_multitask_fromsmiles_dm(
        self,
    ):  # TODO: I think we can remove this as it tests tiny_zinc which only contain graph level labels
        """Cover similar testing as for the original data module."""
        df = graphium.data.load_tiny_zinc()  # 100 molecules

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score' in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[["SMILES", "SA"]]
        df_micro_zinc_logp = df[["SMILES", "logp"]]
        df_micro_zinc_score = df[["SMILES", "score"]]

        # Setup the featurization. This will be the same across all tasks.
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["in-ring", "bond-type-onehot"]
        featurization_args["add_self_loop"] = False
        featurization_args["use_bonds_weights"] = False
        featurization_args["explicit_H"] = False

        # Config for multitask datamodule.

        # Per-task arguments.
        dm_task_args_SA = {}
        dm_task_args_SA["df"] = df_micro_zinc_SA
        dm_task_args_SA["task_level"] = "graph"
        dm_task_args_SA["smiles_col"] = "SMILES"
        dm_task_args_SA["label_cols"] = ["SA"]
        dm_task_args_SA["split_val"] = 0.2
        dm_task_args_SA["split_test"] = 0.2
        dm_task_args_SA["seed"] = 19
        dm_task_args_SA["splits_path"] = None  # This may not always be provided
        dm_task_args_SA["sample_size"] = None  # This may not always be provided
        dm_task_args_SA["idx_col"] = None  # This may not always be provided
        dm_task_args_SA["weights_col"] = None  # This may not always be provided
        dm_task_args_SA["weights_type"] = None  # This may not always be provided

        dm_task_args_logp = {}
        dm_task_args_logp["df"] = df_micro_zinc_logp
        dm_task_args_logp["task_level"] = "graph"
        dm_task_args_logp["smiles_col"] = "SMILES"
        dm_task_args_logp["label_cols"] = ["logp"]
        dm_task_args_logp["split_val"] = 0.2
        dm_task_args_logp["split_test"] = 0.2
        dm_task_args_logp["seed"] = 19
        dm_task_args_logp["splits_path"] = None  # This may not always be provided
        dm_task_args_logp["sample_size"] = None  # This may not always be provided
        dm_task_args_logp["idx_col"] = None  # This may not always be provided
        dm_task_args_logp["weights_col"] = None  # This may not always be provided
        dm_task_args_logp["weights_type"] = None  # This may not always be provided

        dm_task_args_score = {}
        dm_task_args_score["df"] = df_micro_zinc_score
        dm_task_args_score["task_level"] = "graph"
        dm_task_args_score["smiles_col"] = "SMILES"
        dm_task_args_score["label_cols"] = ["score"]
        dm_task_args_score["split_val"] = 0.2
        dm_task_args_score["split_test"] = 0.2
        dm_task_args_score["seed"] = 19
        dm_task_args_score["splits_path"] = None  # This may not always be provided
        dm_task_args_score["sample_size"] = None  # This may not always be provided
        dm_task_args_score["idx_col"] = None  # This may not always be provided
        dm_task_args_score["weights_col"] = None  # This may not always be provided
        dm_task_args_score["weights_type"] = None  # This may not always be provided

        dm_task_kwargs = {}
        dm_task_kwargs["SA"] = dm_task_args_SA
        dm_task_kwargs["logp"] = dm_task_args_logp
        dm_task_kwargs["score"] = dm_task_args_score

        dm_args = {}

        # Task-specific arguments for the datamodule
        dm_args["task_specific_args"] = dm_task_kwargs

        # Task-independent arguments
        dm_args["featurization"] = featurization_args
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True
        dm_args["featurization_backend"] = "loky"
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["processed_graph_data_path"] = None
        dm_args["batch_size_training"] = 16
        dm_args["batch_size_inference"] = 16

        # Create the data module
        dm = graphium.data.MultitaskFromSmilesDataModule(**dm_args)

        # self.assertEqual(50, dm.num_node_feats)    # Not implemeneted error
        # self.assertEqual(6, dm.num_edge_feats)

        dm.prepare_data()
        dm.setup()

        # self.assertEqual(len(dm), 100)                      # Should this have a fixed value for when it's initialized? MTL dataset only gets created after.
        self.assertEqual(len(dm.train_ds), 60)  # type: ignore
        self.assertEqual(len(dm.val_ds), 20)  # type: ignore
        self.assertEqual(len(dm.test_ds), 20)  # type: ignore
        # assert dm.num_node_feats == 50
        # assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)

            assert set(batch.keys()) == {"labels", "features"}

            # assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["graph_SA"].shape == (16, 1)
            assert batch["labels"]["graph_logp"].shape == (16, 1)
            assert batch["labels"]["graph_score"].shape == (16, 1)

    # @pytest.mark.skip
    def test_multitask_fromsmiles_from_config(self):
        config = graphium.load_config(name="zinc_default_multitask_pyg")

        df = graphium.data.load_tiny_zinc()  # 100 molecules

        # Here we take the microzinc dataset and split the labels up into 'SA', 'logp' and 'score' in order to simulate having multiple single-task datasets
        df_micro_zinc_SA = df[["SMILES", "SA"]]
        df_micro_zinc_logp = df[["SMILES", "logp"]]
        df_micro_zinc_score = df[["SMILES", "score"]]

        # dm_args = dict(config.datamodule.args)
        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        # dm_args["task_specific_args"]["SA"]["df"] = df
        dm_args["task_specific_args"]["SA"]["df"] = df_micro_zinc_SA
        dm_args["task_specific_args"]["logp"]["df"] = df_micro_zinc_logp
        dm_args["task_specific_args"]["score"]["df"] = df_micro_zinc_score

        dm_args["task_specific_args"]["SA"]["smiles_col"] = "SMILES"
        dm_args["task_specific_args"]["logp"]["smiles_col"] = "SMILES"
        dm_args["task_specific_args"]["score"]["smiles_col"] = "SMILES"

        dm_args["task_specific_args"]["SA"]["label_cols"] = ["SA"]
        dm_args["task_specific_args"]["logp"]["label_cols"] = ["logp"]
        dm_args["task_specific_args"]["score"]["label_cols"] = ["score"]

        dm_args["task_specific_args"]["SA"]["df_path"] = None
        dm_args["task_specific_args"]["logp"]["df_path"] = None
        dm_args["task_specific_args"]["score"]["df_path"] = None

        dm = graphium.data.MultitaskFromSmilesDataModule(**dm_args)

        # assert dm.num_node_feats == 50
        # assert dm.num_edge_feats == 6

        dm.prepare_data()
        dm.setup()

        # self.assertEqual(len(dm), 100)                      # Should this have a fixed value for when it's initialized? MTL dataset only gets created after.
        self.assertEqual(len(dm.train_ds), 60)  # type: ignore
        self.assertEqual(len(dm.val_ds), 20)  # type: ignore
        self.assertEqual(len(dm.test_ds), 20)  # type: ignore
        # assert dm.num_node_feats == 50
        # assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)

            assert set(batch.keys()) == {"labels", "features"}

            # assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["graph_SA"].shape == (16, 1)
            assert batch["labels"]["graph_logp"].shape == (16, 1)
            assert batch["labels"]["graph_score"].shape == (16, 1)

    def test_multitask_fromsmiles_from_config_csv(self):
        config = graphium.load_config(name="zinc_default_multitask_pyg")

        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        dm = graphium.data.MultitaskFromSmilesDataModule(**dm_args)

        dm.prepare_data()
        dm.setup()

        # self.assertEqual(len(dm), 100)                      # Should this have a fixed value for when it's initialized? MTL dataset only gets created after.
        self.assertEqual(len(dm.train_ds), 60)  # type: ignore
        self.assertEqual(len(dm.val_ds), 20)  # type: ignore
        self.assertEqual(len(dm.test_ds), 20)  # type: ignore
        # assert dm.num_node_feats == 50
        # assert dm.num_edge_feats == 6

        for dl in [dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()]:
            it = iter(dl)
            batch = next(it)

            assert set(batch.keys()) == {"labels", "features"}

            # assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["graph_SA"].shape == (16, 1)
            assert batch["labels"]["graph_logp"].shape == (16, 1)
            assert batch["labels"]["graph_score"].shape == (16, 1)

    def test_multitask_fromsmiles_from_config_parquet(self):
        config = graphium.load_config(name="fake_multilevel_multitask_pyg")

        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        dm = graphium.data.MultitaskFromSmilesDataModule(**dm_args)

        dm.prepare_data()
        dm.setup()

        self.assertEqual(len(dm.train_ds), 1004)  # type: ignore

        dl = dm.train_dataloader()
        it = iter(dl)
        batch = next(it)

        assert set(batch.keys()) == {"labels", "features"}

        # assert batch["labels"].shape == (16, 1)            # Single-task case
        assert batch["labels"]["graph_SA"].shape == (16, 1)
        assert batch["labels"]["node_logp"].shape == (
            batch["features"].feat.size(0),
            2,
        )  # test node level
        assert batch["labels"]["edge_score"].shape == (
            batch["features"].edge_feat.size(0),
            2,
        )  # test edge level

    def test_multitask_with_missing_fromsmiles_from_config_parquet(self):
        config = graphium.load_config(name="fake_and_missing_multilevel_multitask_pyg")

        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        dm = graphium.data.MultitaskFromSmilesDataModule(**dm_args)

        dm.prepare_data()
        dm.setup()

        self.assertEqual(len(dm.train_ds), 1004)  # type: ignore

        dl = dm.train_dataloader()
        it = iter(dl)
        batch = next(it)

        assert set(batch.keys()) == {"labels", "features"}

        # assert batch["labels"].shape == (16, 1)            # Single-task case
        assert batch["labels"]["graph_SA"].shape == (16, 1)
        assert batch["labels"]["node_logp"].shape == (
            batch["features"].feat.size(0),
            2,
        )  # test node level
        assert batch["labels"]["edge_score"].shape == (
            batch["features"].edge_feat.size(0),
            2,
        )  # test edge level

    def test_extract_graph_level_singletask(self):
        df = pd.read_parquet(f"tests/converted_fake_multilevel_data.parquet")
        num_graphs = len(df)
        label_cols = ["graph_label"]
        output = graphium.data.datamodule.extract_labels(df, "graph", label_cols)

        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 2
        assert output.shape[0] == num_graphs
        assert output.shape[1] == 1

    def test_extract_graph_level_multitask(self):
        df = pd.read_parquet(f"tests/converted_fake_multilevel_data.parquet")
        num_graphs = len(df)
        label_cols = ["graph_label", "graph_label"]
        output = graphium.data.datamodule.extract_labels(df, "graph", label_cols)

        assert isinstance(output, np.ndarray)
        assert len(output.shape) == 2
        assert output.shape[0] == num_graphs
        assert output.shape[1] == len(label_cols)

    def test_extract_graph_level_multitask_missing_cols(self):
        df = pd.read_parquet(f"tests/converted_fake_multilevel_data.parquet")
        num_graphs = len(df)
        label_cols = ["graph_label", "graph_label"]

        drop_index = [2, 5, 21, 237, 192, 23, 127, 11]
        for replace in [1, 2]:
            for missing_col in label_cols[:replace]:
                df[missing_col].iloc[drop_index] = None

            output = graphium.data.datamodule.extract_labels(df, "graph", label_cols)

            assert isinstance(output, np.ndarray)
            assert len(output.shape) == 2
            assert output.shape[0] == num_graphs
            assert output.shape[1] == len(label_cols)

    def test_non_graph_level_extract_labels(self):
        df = pd.read_parquet(f"tests/converted_fake_multilevel_data.parquet")

        for level in ["node", "edge", "nodepair"]:
            label_cols = [f"{level}_label_{suffix}" for suffix in ["list", "np"]]
            output = graphium.data.datamodule.extract_labels(df, level, label_cols)

            assert isinstance(output, list)
            assert len(output[0].shape) == 2
            assert output[0].shape[1] == len(label_cols)

    def test_non_graph_level_extract_labels_missing_cols(self):
        df = pd.read_parquet(f"tests/converted_fake_multilevel_data.parquet")

        for level in ["node", "edge", "nodepair"]:
            label_cols = [f"{level}_label_{suffix}" for suffix in ["list", "np"]]
            drop_index = [2, 5, 21, 237, 192, 23, 127, 11]
            for replace in [1, 2]:
                for missing_col in label_cols[:replace]:
                    df.loc[drop_index, missing_col] = None

                output = graphium.data.datamodule.extract_labels(df, level, label_cols)

                for idx in drop_index:
                    assert len(output[idx].shape) == 2
                    assert output[idx].shape[1] == len(label_cols)

                    # Check that number of labels is adjusted correctly
                    if replace == 1:
                        non_missing_col = label_cols[1]
                        assert output[idx].shape[0] == len(df[non_missing_col][idx])

    def test_tdc_admet_benchmark_data_module(self):
        """
        Verifies that the ADMET-specific subclass of the MultiTaskDataModule works.
        Checks if all main endpoints can be run and if the split is correct.
        """

        try:
            from tdc.benchmark_group import admet_group
            from tdc.utils import retrieve_benchmark_names
        except ImportError:
            self.skipTest("PyTDC needs to be installed to run this test. Use `pip install PyTDC`.")
            raise

        # Make sure we can initialize the module and run the main endpoints
        data_module = graphium.data.ADMETBenchmarkDataModule()
        data_module.prepare_data()
        data_module.setup()

        for dl in [
            data_module.train_dataloader(),
            data_module.val_dataloader(),
            data_module.test_dataloader(),
        ]:
            batch = next(iter(dl))
            assert set(batch.keys()) == {"labels", "features"}

        # # Validate the split
        group = admet_group(path=self.tmp_test_dir)
        benchmark_names = retrieve_benchmark_names("admet_group")

        # For each of the endpoints...
        for name in benchmark_names:
            # Get the split from the benchmark group (ground truth)
            benchmark = group.get(name)
            train, val = group.get_train_valid_split(0, name)
            test = benchmark["test"]

            # Get the split from the data module
            params = data_module._get_task_specific_arguments(name, 0, self.tmp_test_dir)
            split = pd.read_csv(params.splits_path)
            data = params.df

            # Check that the split is the same
            for ground_truth, label in [(train, "train"), (val, "val"), (test, "test")]:
                y_true = ground_truth["Y"].values
                y_module = data.loc[split[label].dropna(), "Y"].values

                assert len(y_true) == len(y_module)
                assert np.allclose(np.sort(y_true), np.sort(y_module))


if __name__ == "__main__":
    ut.main()
