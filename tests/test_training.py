import pytest
from graphium.cli.train_finetune_test import cli
import sys


def call_cli_with_overrides(acc_type: str, acc_prec: str, load_type: str) -> None:
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
def test_cpu_cli_training(load_type):
    call_cli_with_overrides("cpu", "32", load_type)


@pytest.mark.ipu
@pytest.mark.parametrize("load_type", ["RAM", "disk"])
def test_ipu_cli_training(load_type):
    call_cli_with_overrides("ipu", "16-true", load_type)
