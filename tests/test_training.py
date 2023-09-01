import pytest
from graphium.cli.train_finetune_test import cli
import sys
import subprocess
import os
from unittest.mock import patch


class TestCLITraining:
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

    def call_cli_with_overrides(self, acc_type: str, acc_prec: str, load_type: str) -> None:
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
            f"datamodule.args.dataloading_from={load_type}",
        ]
        if acc_type == "ipu":
            overrides.append("accelerator.ipu_config=['useIpuModel(True)']")
            overrides.append("accelerator.ipu_inference_config=['useIpuModel(True)']")
        # Backup the original sys.argv
        original_argv = sys.argv.copy()

        # Replace sys.argv with the desired overrides
        hydra_overrides = ["script_name"] + overrides
        sys.argv = hydra_overrides
        # Call the function
        cli()

        # Restore the original sys.argv
        sys.argv = original_argv

    @pytest.mark.parametrize("load_type", ["RAM", "disk"])
    def test_cpu_cli_training(self, load_type):
        self.call_cli_with_overrides("cpu", "32", load_type)

    @pytest.mark.ipu
    @pytest.mark.skip
    @pytest.mark.parametrize("load_type", ["RAM", "disk"])
    def test_ipu_cli_training(self, load_type):
        with patch("poptorch.ipuHardwareIsAvailable", return_value=True):
            with patch("lightning_graphcore.accelerator._IPU_AVAILABLE", new=True):
                import poptorch

                assert poptorch.ipuHardwareIsAvailable()
                from lightning_graphcore.accelerator import _IPU_AVAILABLE

                assert _IPU_AVAILABLE is True
                self.call_cli_with_overrides("ipu", "16-true", load_type)
