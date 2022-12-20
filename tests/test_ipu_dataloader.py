# General imports
import yaml
import unittest as ut
import numpy as np
from copy import deepcopy
from warnings import warn
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.plugins import IPUPlugin
from functools import partial

import torch
from torch.utils.data.dataloader import default_collate

# Current library imports
from goli.ipu.ipu_dataloader import smart_packing, get_pack_sizes, fast_packing, hybrid_packing
from goli.ipu.ipu_wrapper import PredictorModuleIPU
from goli.config._loader import load_datamodule, load_metrics, load_architecture
from goli.ipu.ipu_wrapper import IPUPluginGoli


def random_packing(num_nodes, batch_size):
    ipu_batch_size = int(len(num_nodes) / batch_size)
    indices = np.arange(len(num_nodes))
    np.random.shuffle(indices)
    indices = np.reshape(indices, (ipu_batch_size, batch_size)).tolist()
    return indices


def global_batch_collator(batch_size, batches):
    packs = []
    for pack_idx in range(0, len(batches), batch_size):
        packs.append(default_collate(batches[pack_idx : pack_idx + batch_size]))
    global_batch = default_collate(packs)
    global_batch = (global_batch[0], tuple(global_batch[1]))
    return global_batch


class test_Packing(ut.TestCase):
    def test_smart_packing(self):

        np.random.seed(42)

        batch_sizes = [2, 4, 8, 16, 32, 64]
        ipu_batch_sizes = [2, 3, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:

                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = smart_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_fast_packing(self):

        np.random.seed(42)

        # Start at 4 for fast_packing for better statistical significance
        batch_sizes = [4, 8, 16, 32, 64]
        ipu_batch_sizes = [4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:

                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = fast_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )

    def test_hybrid_packing(self):

        np.random.seed(42)

        batch_sizes = [2, 4, 8, 16, 32, 64]
        ipu_batch_sizes = [2, 3, 4, 8, 16, 32, 64]

        for batch_size in batch_sizes:
            for ipu_batch_size in ipu_batch_sizes:

                err_msg = f"bz={batch_size}, ipu_bz={ipu_batch_size}"

                # Generate random batch size
                global_batch = batch_size * ipu_batch_size
                num_nodes = np.abs(np.random.gamma(2, 20, size=global_batch)).astype(int)

                # Use the smart packing
                packed_indices = hybrid_packing(num_nodes=num_nodes, batch_size=batch_size)
                pack_num_nodes = get_pack_sizes(packed_indices, num_nodes)

                # Use the random packing
                rand_packed_indices = random_packing(num_nodes=num_nodes, batch_size=batch_size)
                rand_pack_num_nodes = get_pack_sizes(rand_packed_indices, num_nodes)

                # Assert that the smart packing is better than the random packing
                self.assertLessEqual(max(pack_num_nodes), max(rand_pack_num_nodes), msg=err_msg)
                self.assertGreaterEqual(min(pack_num_nodes), min(rand_pack_num_nodes), msg=err_msg)

                # Assert that the total number of atoms is right
                self.assertEqual(sum(pack_num_nodes), sum(num_nodes), msg=err_msg)
                self.assertEqual(sum(rand_pack_num_nodes), sum(num_nodes), msg=err_msg)

                # Assert that all index are there
                self.assertListEqual(
                    np.sort(np.asarray(packed_indices).flatten()).tolist(), np.arange(len(num_nodes)).tolist()
                )
                self.assertListEqual(
                    np.sort(np.asarray(rand_packed_indices).flatten()).tolist(),
                    np.arange(len(num_nodes)).tolist(),
                )


class test_DataLoading(ut.TestCase):
    class TestSimpleLightning(LightningModule):
        # Create a basic Ligthning for testing the batch sizes
        def __init__(self, batch_size, node_feat_size, edge_feat_size, num_batch) -> None:
            super().__init__()
            self.batch_size = batch_size
            self.node_feat_size = node_feat_size
            self.edge_feat_size = edge_feat_size
            self.layer = torch.nn.Linear(node_feat_size, 1)
            self.loss_fn = torch.nn.L1Loss()
            self.num_batch = num_batch

        def validation_step(self, batch, batch_idx):
            self.assert_shapes(batch, batch_idx, "val")
            loss = self.forward(batch)
            return loss

        def training_step(self, batch, batch_idx):
            self.assert_shapes(batch, batch_idx, "train")
            loss = self.forward(batch)
            return loss

        def forward(self, batch):
            out = self.layer(batch[1][0]).squeeze(-1)
            loss = self.loss_fn(out, batch[0])
            return loss

        def assert_shapes(self, batch, batch_idx, step):
            msg = f"\nbatch_idx=`{batch_idx}`, step=`{step}`"

            # Test the shape of the labels
            this_shape = list(batch[0].shape)
            true_shape = [1, self.batch_size]
            assert (
                this_shape == true_shape
            ), f"Shape of the labels is `{this_shape}` but should be {true_shape}. {msg}"

            # Test the shape of the first feature
            this_shape = list(batch[1][0].shape)
            true_shape = [1, self.batch_size, self.node_feat_size]
            assert (
                this_shape == true_shape
            ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}. {msg}"

            # Test the shape of the second feature
            this_shape = list(batch[1][1].shape)
            true_shape = [1, self.batch_size, self.edge_feat_size]
            assert (
                this_shape == true_shape
            ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}. {msg}"

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    class TestPredictor(PredictorModuleIPU):
        # Create a basic Ligthning for testing the batch sizes
        def __init__(self, batch_size, node_batch_size, edge_batch_size, in_dims, **kwargs) -> None:
            super().__init__(**kwargs)
            self.in_dims = in_dims
            self.batch_size = batch_size
            self.node_batch_size = node_batch_size
            self.edge_batch_size = edge_batch_size

            import poptorch

            self.pop = poptorch

        def validation_step(self, *args):
            dict_input = self._build_dict_input(*args)
            self.assert_shapes(args, dict_input, "val")
            preds = self.forward(dict_input)["preds"]
            loss = self.compute_loss(
                preds, dict_input["labels"], weights=None, loss_fun=self.loss_fun, target_nan_mask=None
            )
            loss = loss[0]
            return loss

        def training_step(self, *args):
            dict_input = self._build_dict_input(*args)
            self.assert_shapes(args, dict_input, "train")
            preds = self.forward(dict_input)["preds"]
            loss = self.compute_loss(
                preds, dict_input["labels"], weights=None, loss_fun=self.loss_fun, target_nan_mask=None
            )
            loss = self.poptorch.identity_loss(loss[0], reduction="mean")
            return loss

        def assert_shapes(self, args, dict_input, step):
            msg = f", step=`{step}`"

            # Ensure that the first dimension is 1 for all tensors
            for arg in args:
                assert arg.shape[0] == 1

            # Test the shape of the labels
            this_shape = list(dict_input["labels"]["homo"].shape)
            true_shape = [self.batch_size, 2]
            assert (
                this_shape == true_shape
            ), f"Shape of the labels is `{this_shape}` but should be {true_shape}. {msg}"

            # Test the shape of the node feature
            this_shape = list(dict_input["features"]["feat"].shape)
            true_shape = [self.node_batch_size, self.in_dims["feat"]]
            assert (
                this_shape == true_shape
            ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}. {msg}"

            # Test the shape of the node feature
            this_shape = list(dict_input["features"]["edge_feat"].shape)
            true_shape = [self.edge_batch_size, self.in_dims["edge_feat"]]
            assert (
                this_shape == true_shape
            ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}. {msg}"

        def get_progress_bar_dict(self):
            return {}

        def on_train_batch_end(self, *args, **kwargs):
            return

        def on_validation_batch_end(self, *args, **kwargs):
            return

        def validation_epoch_end(self, *args, **kwargs):
            return

        def on_train_epoch_end(self) -> None:
            return

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    class TestDataset(torch.utils.data.Dataset):
        # Create a simple dataset for testing the Lightning integration
        def __init__(self, labels, node_features, edge_features):
            self.labels = labels
            self.node_features = node_features
            self.edge_features = edge_features

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            # [label, [feat1, feat2]]
            return [self.labels[idx], [self.node_features[idx], self.edge_features[idx]]]

    def test_poptorch_simple_deviceiterations_gradient_accumulation(self):
        """
        Test a simple version of the device-iterations and gradient accumulation
        to make sure that the dataloader and models handle them correcly.
        """

        # Run this test only if poptorch is available
        try:
            import poptorch
        except Exception as e:
            warn(f"Skipping this test because poptorch is not available.\n{e}")
            return

        # Initialize constants
        gradient_accumulation = 2
        device_iterations = 3
        batch_size = 5
        num_replicate = 7
        node_feat_size = 11
        edge_feat_size = 13

        # Initialize the batch info and poptorch options
        opts = poptorch.Options()
        opts.deviceIterations(device_iterations)
        opts.Jit.traceModel(True)
        training_opts = deepcopy(opts)
        training_opts.Training.gradientAccumulation(gradient_accumulation)
        inference_opts = deepcopy(opts)

        # Initialize the dataset
        num_batch = device_iterations * gradient_accumulation * num_replicate
        data_size = num_batch * batch_size
        dataset = self.TestDataset(
            labels=np.random.rand(data_size).astype(np.float32),
            node_features=[np.random.rand(node_feat_size).astype(np.float32) for ii in range(data_size)],
            edge_features=[np.random.rand(edge_feat_size).astype(np.float32) for ii in range(data_size)],
        )

        # Initialize the dataloader
        train_dataloader = poptorch.DataLoader(
            options=training_opts,
            dataset=deepcopy(dataset),
            batch_size=batch_size,
            collate_fn=partial(global_batch_collator, batch_size),
        )

        val_dataloader = poptorch.DataLoader(
            options=inference_opts,
            dataset=deepcopy(dataset),
            batch_size=batch_size,
            collate_fn=partial(global_batch_collator, batch_size),
        )

        # Build the model, and run it on IPU
        model = self.TestSimpleLightning(batch_size, node_feat_size, edge_feat_size, num_batch)
        plugins = IPUPlugin(training_opts=training_opts, inference_opts=inference_opts)
        trainer = Trainer(
            logger=False, enable_checkpointing=False, max_epochs=2, plugins=plugins, num_sanity_val_steps=0
        )
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    def test_poptorch_goli_deviceiterations_gradient_accumulation(self):
        """
        Test the device-iterations and gradient accumulation in a way
        that is very similar to the Goli code
        to make sure that the dataloader and models handle them correcly.
        """

        try:
            import poptorch
        except Exception as e:
            warn(f"Skipping this test because poptorch is not available.\n{e}")
            return

        gradient_accumulation = 3
        device_iterations = 5
        batch_size = 7

        # Initialize the batch info and poptorch options
        opts = poptorch.Options()
        opts.deviceIterations(device_iterations)
        opts.Jit.traceModel(True)
        training_opts = deepcopy(opts)
        training_opts.Training.gradientAccumulation(gradient_accumulation)
        inference_opts = deepcopy(opts)

        # Load the configuration file for the model
        CONFIG_FILE = "tests/config_test_ipu_dataloader.yaml"
        with open(CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["datamodule"]["args"]["batch_size_training"] = batch_size
        cfg["datamodule"]["args"]["batch_size_inference"] = batch_size
        node_factor = cfg["datamodule"]["args"]["ipu_dataloader_training_opts"]["max_num_nodes_per_graph"]
        edge_factor = cfg["datamodule"]["args"]["ipu_dataloader_training_opts"]["max_num_edges_per_graph"]

        # Load the datamodule, and prepare the data
        datamodule = load_datamodule(cfg)
        datamodule.prepare_data()
        datamodule.setup()
        datamodule.ipu_training_opts = training_opts
        datamodule.ipu_inference_opts = inference_opts

        model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)
        metrics = load_metrics(cfg)
        predictor = self.TestPredictor(
            batch_size=batch_size,
            node_batch_size=node_factor * batch_size,
            edge_batch_size=edge_factor * batch_size,
            in_dims=datamodule.in_dims,
            model_class=model_class,
            model_kwargs=model_kwargs,
            metrics=metrics,
            **cfg["predictor"],
        )
        plugins = IPUPluginGoli(training_opts=training_opts, inference_opts=inference_opts)
        trainer = Trainer(
            logger=False, enable_checkpointing=False, max_epochs=2, plugins=plugins, num_sanity_val_steps=0
        )
        trainer.fit(model=predictor, datamodule=datamodule)


if __name__ == "__main__":
    ut.main()
