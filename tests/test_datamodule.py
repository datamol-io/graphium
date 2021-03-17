import goli
import unittest as ut


class Test_DataModule(ut.TestCase):
    def test_dglfromsmiles_dm(self):

        # NOTE(hadim): we could parametrized the test in order to test
        # different scenario.

        df = goli.data.load_tiny_zinc()
        # Setup the featurization
        featurization_args = {}
        featurization_args["atom_property_list_float"] = []  # ["weight", "valence"]
        featurization_args["atom_property_list_onehot"] = ["atomic-number", "degree"]
        featurization_args["edge_property_list"] = ["ring", "bond-type-onehot"]
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
        dm_args["train_val_batch_size"] = 16
        dm_args["test_batch_size"] = 16
        dm_args["num_workers"] = 0
        dm_args["pin_memory"] = True
        dm_args["featurization_n_jobs"] = 16
        dm_args["featurization_progress"] = True

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

        config = goli.load_config(name="micro_zinc_default")
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


if __name__ == "__main__":
    ut.main()
