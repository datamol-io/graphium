# General imports
import yaml
import unittest as ut
import numpy as np
from copy import deepcopy
from warnings import warn
from unittest.mock import patch
from lightning import Trainer, LightningModule
from functools import partial
import pytest
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch.utils.data.dataloader import default_collate
from lightning_graphcore import IPUStrategy


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


@pytest.mark.ipu
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
            # Test the shape of the labels
            this_shape = list(batch[0].shape)
            true_shape = [1, self.batch_size]
            assert (
                this_shape == true_shape
            ), f"Shape of the labels is `{this_shape}` but should be {true_shape}"

            # Test the shape of the first feature
            this_shape = list(batch[1][0].shape)
            true_shape = [1, self.batch_size, self.node_feat_size]
            assert (
                this_shape == true_shape
            ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}"

            # Test the shape of the second feature
            this_shape = list(batch[1][1].shape)
            true_shape = [1, self.batch_size, self.edge_feat_size]
            assert (
                this_shape == true_shape
            ), f"Shape of the feature 0 is `{this_shape}` but should be {true_shape}"

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

    # @pytest.mark.skip
    def test_poptorch_simple_deviceiterations_gradient_accumulation(self):
        """
        Test a simple version of the device-iterations and gradient accumulation
        to make sure that the dataloader and models handle them correcly.
        """

        with patch("poptorch.ipuHardwareIsAvailable", return_value=True):
            with patch("lightning_graphcore.accelerator._IPU_AVAILABLE", new=True):
                import poptorch

                assert poptorch.ipuHardwareIsAvailable()
                from lightning_graphcore.accelerator import _IPU_AVAILABLE

                assert _IPU_AVAILABLE is True

                # Initialize constants
                gradient_accumulation = 2
                device_iterations = 3
                batch_size = 5
                num_replicate = 7
                node_feat_size = 11
                edge_feat_size = 13

                # Initialize the batch info and poptorch options
                opts = poptorch.Options()
                opts.useIpuModel(True)
                opts.deviceIterations(device_iterations)
                training_opts = deepcopy(opts)
                training_opts.Training.gradientAccumulation(gradient_accumulation)
                inference_opts = deepcopy(opts)

                # Initialize the dataset
                num_batch = device_iterations * gradient_accumulation * num_replicate
                data_size = num_batch * batch_size
                dataset = self.TestDataset(
                    labels=np.random.rand(data_size).astype(np.float32),
                    node_features=[
                        np.random.rand(node_feat_size).astype(np.float32) for ii in range(data_size)
                    ],
                    edge_features=[
                        np.random.rand(edge_feat_size).astype(np.float32) for ii in range(data_size)
                    ],
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

                # Build the model, and run it on "IPU"
                model = self.TestSimpleLightning(batch_size, node_feat_size, edge_feat_size, num_batch)

                strategy = IPUStrategy(
                    training_opts=training_opts, inference_opts=inference_opts, autoreport=True
                )
                trainer = Trainer(
                    logger=True,
                    enable_checkpointing=False,
                    max_epochs=2,
                    strategy=strategy,
                    num_sanity_val_steps=0,
                    accelerator="ipu",
                    devices=1,
                )
                trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    @pytest.mark.skip
    def test_poptorch_graphium_deviceiterations_gradient_accumulation_full(self):
        """
        Test the device-iterations and gradient accumulation in a way
        that is very similar to the Graphium code
        to make sure that the dataloader and models handle them correcly.
        """
        with patch("poptorch.ipuHardwareIsAvailable", return_value=True):
            with patch("lightning_graphcore.accelerator._IPU_AVAILABLE", new=True):
                try:
                    import poptorch
                except Exception as e:
                    warn(f"Skipping this test because poptorch is not available.\n{e}")
                    return

                from lightning_graphcore import IPUStrategy
                import lightning_graphcore

                # Current library imports
                from graphium.config._loader import (
                    load_datamodule,
                    load_metrics,
                    load_architecture,
                    load_accelerator,
                    load_predictor,
                    load_trainer,
                )
                from graphium.utils.safe_run import SafeRun

                # Simplified testing config - reflecting the toymix requirements
                CONFIG_FILE = "tests/config_test_ipu_dataloader_multitask.yaml"
                with open(CONFIG_FILE, "r") as f:
                    cfg = yaml.safe_load(f)

                cfg, accelerator = load_accelerator(cfg)

                # Load the datamodule, and prepare the data
                datamodule = load_datamodule(cfg, accelerator_type=accelerator)
                datamodule.prepare_data()
                metrics = load_metrics(cfg)
                model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)
                # datamodule.setup()
                predictor = load_predictor(
                    cfg,
                    model_class,
                    model_kwargs,
                    metrics,
                    datamodule.get_task_levels(),
                    accelerator,
                    datamodule.featurization,
                    datamodule.task_norms,
                )
                assert poptorch.ipuHardwareIsAvailable()
                trainer = load_trainer(cfg, "test", accelerator, "date_time_suffix")
                # Run the model training
                with SafeRun(
                    name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True
                ):
                    trainer.fit(model=predictor, datamodule=datamodule)


if __name__ == "__main__":
    ut.main()
