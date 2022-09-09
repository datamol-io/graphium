import unittest as ut
from omegaconf import OmegaConf

import goli


class Test_Multitask_DataModule(ut.TestCase):
    def test_multitask_fromsmiles_dm(self):
        """Cover similar testing as for the original data module."""
        df = goli.data.load_tiny_zinc()  # 100 molecules

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
        dm_task_args_SA["smiles_col"] = "SMILES"
        dm_task_args_SA["label_cols"] = ["SA"]
        dm_task_args_SA["split_val"] = 0.2
        dm_task_args_SA["split_test"] = 0.2
        dm_task_args_SA["split_seed"] = 19
        dm_task_args_SA["splits_path"] = None  # This may not always be provided
        dm_task_args_SA["sample_size"] = None  # This may not always be provided
        dm_task_args_SA["idx_col"] = None  # This may not always be provided
        dm_task_args_SA["weights_col"] = None  # This may not always be provided
        dm_task_args_SA["weights_type"] = None  # This may not always be provided

        dm_task_args_logp = {}
        dm_task_args_logp["df"] = df_micro_zinc_logp
        dm_task_args_logp["smiles_col"] = "SMILES"
        dm_task_args_logp["label_cols"] = ["logp"]
        dm_task_args_logp["split_val"] = 0.2
        dm_task_args_logp["split_test"] = 0.2
        dm_task_args_logp["split_seed"] = 19
        dm_task_args_logp["splits_path"] = None  # This may not always be provided
        dm_task_args_logp["sample_size"] = None  # This may not always be provided
        dm_task_args_logp["idx_col"] = None  # This may not always be provided
        dm_task_args_logp["weights_col"] = None  # This may not always be provided
        dm_task_args_logp["weights_type"] = None  # This may not always be provided

        dm_task_args_score = {}
        dm_task_args_score["df"] = df_micro_zinc_score
        dm_task_args_score["smiles_col"] = "SMILES"
        dm_task_args_score["label_cols"] = ["score"]
        dm_task_args_score["split_val"] = 0.2
        dm_task_args_score["split_test"] = 0.2
        dm_task_args_score["split_seed"] = 19
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
        dm_args["featurization_backend"] = "threads"
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["cache_data_path"] = None
        dm_args["batch_size_train_val"] = 16
        dm_args["batch_size_test"] = 16

        # Create the data module
        dm = goli.data.MultitaskFromSmilesDataModule(**dm_args)

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

            assert set(batch.keys()) == {"labels", "features", "smiles", "mol_ids"}
            assert len(batch["mol_ids"]) == 16
            # assert len(batch["smiles"]) == 16
            # assert len(batch["features"])                      # This is not a list, but a graph.

            # assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["SA"].shape == (16, 1)
            assert batch["labels"]["logp"].shape == (16, 1)
            assert batch["labels"]["score"].shape == (16, 1)

    def test_multitask_fromsmiles_from_config(self):

        config = goli.load_config(name="zinc_default_multitask_fulldgl")

        df = goli.data.load_tiny_zinc()  # 100 molecules

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

        dm_args["task_specific_args"]["SA"]["df_path"] = None
        dm_args["task_specific_args"]["logp"]["df_path"] = None
        dm_args["task_specific_args"]["score"]["df_path"] = None

        dm = goli.data.MultitaskFromSmilesDataModule(**dm_args)

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

            assert set(batch.keys()) == {"labels", "features", "smiles", "mol_ids"}
            assert len(batch["mol_ids"]) == 16
            # assert len(batch["smiles"]) == 16
            # assert len(batch["features"])                      # This is not a list, but a graph.

            # assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["SA"].shape == (16, 1)
            assert batch["labels"]["logp"].shape == (16, 1)
            assert batch["labels"]["score"].shape == (16, 1)

    def test_multitask_fromsmiles_from_config_csv(self):
        config = goli.load_config(name="zinc_default_multitask_fulldgl")

        dm_args = OmegaConf.to_container(config.datamodule.args, resolve=True)
        dm = goli.data.MultitaskFromSmilesDataModule(**dm_args)

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

            assert set(batch.keys()) == {"labels", "features", "smiles", "mol_ids"}
            assert len(batch["mol_ids"]) == 16
            # assert len(batch["smiles"]) == 16
            # assert len(batch["features"])                      # This is not a list, but a graph.

            # assert batch["labels"].shape == (16, 1)            # Single-task case
            assert batch["labels"]["SA"].shape == (16, 1)
            assert batch["labels"]["logp"].shape == (16, 1)
            assert batch["labels"]["score"].shape == (16, 1)


if __name__ == "__main__":
    ut.main()
