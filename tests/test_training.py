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


import pytest
from graphium.cli.train_finetune_test import cli
import sys
import subprocess
import os
import shutil
import unittest as ut


import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import os

from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_datamodule,
    load_metrics,
    load_predictor,
    load_trainer,
)

class test_CLITraining():
    @classmethod
    def setup_class(cls):
        print("Setting up the test class...")

        # Equivalent of the bash commands to download the data files
        toymix_dir = "expts/data/neurips2023/small-dataset/"
        subprocess.run(["mkdir", "-p", toymix_dir])

        base_url = "https://storage.googleapis.com/graphium-public/datasets/neurips_2023/Small-dataset/"
        files = [
            "ZINC12k.csv.gz",
            "Tox21-7k-12-labels.csv.gz",
            "qm9.csv.gz",
            "qm9_random_splits.pt",
            "Tox21_random_splits.pt",
            "ZINC12k_random_splits.pt",
        ]

        for file in files:
            file_path = f"{toymix_dir}{file}"
            if not os.path.exists(file_path):
                print(f"Downloading {file}...")
                subprocess.run(["wget", "-P", toymix_dir, f"{base_url}{file}"])
            else:
                print(f"{file} already exists. Skipping...")

        print("Data has been successfully downloaded.")

    def call_cli_with_overrides(self, acc_type: str, acc_prec: str) -> None:
        overrides = [
            f"accelerator={acc_type}",
            "tasks=toymix",
            "training=toymix",
            # Reducing number of parameters in the toymix architecture
            "architecture=toymix",
            "architecture.pe_encoders.encoders.la_pos.hidden_dim=16",
            "architecture.pe_encoders.encoders.la_pos.num_layers=1",
            "architecture.pe_encoders.encoders.rw_pos.hidden_dim=16",
            "architecture.pe_encoders.encoders.rw_pos.num_layers=1",
            "architecture.pre_nn.hidden_dims=32",
            "architecture.pre_nn.depth=1",
            "architecture.pre_nn.out_dim=16",
            "architecture.gnn.in_dim=16",
            "architecture.gnn.out_dim=16",
            "architecture.gnn.depth=2",
            "architecture.task_heads.qm9.depth=1",
            "architecture.task_heads.tox21.depth=1",
            "architecture.task_heads.zinc.depth=1",
            # Set the number of epochs
            "constants.max_epochs=2",
            "+datamodule.args.task_specific_args.qm9.sample_size=1000",
            "+datamodule.args.task_specific_args.tox21.sample_size=1000",
            "+datamodule.args.task_specific_args.zinc.sample_size=1000",
            "trainer.trainer.check_val_every_n_epoch=1",
            f"trainer.trainer.precision={acc_prec}",
        ]

        # Backup the original sys.argv
        original_argv = sys.argv.copy()

        # Replace sys.argv with the desired overrides
        hydra_overrides = ["script_name"] + overrides
        sys.argv = hydra_overrides
        # Call the function
        cli()

        # Restore the original sys.argv
        sys.argv = original_argv

    def test_cpu_cli_training(self):
        self.call_cli_with_overrides("cpu", "32")

    @pytest.mark.ipu
    @pytest.mark.skip
    def test_ipu_cli_training(self):
        with ut.patch("poptorch.ipuHardwareIsAvailable", return_value=True):
            with ut.patch("lightning_graphcore.accelerator._IPU_AVAILABLE", new=True):
                import poptorch

                assert poptorch.ipuHardwareIsAvailable()
                from lightning_graphcore.accelerator import _IPU_AVAILABLE

                assert _IPU_AVAILABLE is True
                self.call_cli_with_overrides("ipu", "16-true")



def initialize_hydra(config_path, job_name="app"):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path=config_path, job_name=job_name)

def compose_main_config(config_dir):
    initialize_hydra(config_dir)
    # Compose the main configuration
    main_config = hydra.compose(config_name="main")
    return main_config

def compose_task_config(config_dir, task_name):
    task_config_dir = os.path.join(config_dir, "tasks")
    initialize_hydra(task_config_dir, job_name="compose_task")
    # Compose the specific task configuration
    task_config = hydra.compose(config_name=task_name)
    return task_config

class test_TrainToymix(ut.TestCase):
    def test_train_toymix(self):
        # Load the main configuration for toymix
        CONFIG_DIR = "../expts/hydra-configs/"
        cfg = compose_main_config(CONFIG_DIR)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg.pop("tasks")

        # Adapt the configuration to reduce the time it takes to run the test, less samples, less epochs
        cfg["constants"]["max_epochs"] = 4
        cfg["trainer"]["trainer"]["check_val_every_n_epoch"] = 1
        cfg["trainer"]["trainer"]["max_epochs"] = 4

        cfg["datamodule"]["args"]["batch_size_training"] = 20
        cfg["datamodule"]["args"]["batch_size_inference"] = 20
        cfg["datamodule"]["args"]["task_specific_args"]["zinc"]["sample_size"] = 300
        cfg["datamodule"]["args"]["task_specific_args"]["qm9"]["sample_size"] = 300
        cfg["datamodule"]["args"]["task_specific_args"]["tox21"]["sample_size"] = 300

        
        # Initialize the accelerator
        cfg, accelerator_type = load_accelerator(cfg)

        # If the data_cache directory exists, delete it for the purpose of the test
        data_cache = cfg["datamodule"]["args"]["processed_graph_data_path"]
        if os.path.exists(data_cache):
            shutil.rmtree(data_cache)

        # Load and initialize the dataset
        datamodule = load_datamodule(cfg, accelerator_type)

        # Initialize the network
        model_class, model_kwargs = load_architecture(
            cfg,
            in_dims=datamodule.in_dims,
        )

        datamodule.prepare_data()

        metrics = load_metrics(cfg)

        predictor = load_predictor(
            cfg,
            model_class,
            model_kwargs,
            metrics,
            datamodule.get_task_levels(),
            accelerator_type,
            datamodule.featurization,
            datamodule.task_norms,
        )

        metrics_on_progress_bar = predictor.get_metrics_on_progress_bar
        trainer = load_trainer(cfg, accelerator_type, metrics_on_progress_bar=metrics_on_progress_bar)

        predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])

        # Run the model training
        trainer.fit(model=predictor, datamodule=datamodule)
        trainer.test(model=predictor, datamodule=datamodule)

if __name__ == "__main__":
    config_dir = "../expts/hydra-configs/"  # Path to your config directory

    ut.main()
