import hydra
from graphium.cli.train_finetune_test import run_training_finetuning_testing
import pytest

FINETUNING_CONFIG_KEY = "finetuning"


@pytest.mark.parametrize("acc_type, acc_prec", [("cpu", 32), ("ipu", 16)])
@pytest.mark.ipu
def test_cli(acc_type, acc_prec) -> None:
    """
    The main CLI endpoint for training and fine-tuning Graphium models.
    """
    with hydra.initialize(version_base=None, config_path="../expts/hydra-configs"):
        # config is relative to a module
        cfg = hydra.compose(
            config_name="main",
            overrides=[
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
                f"trainer.trainer.precision={acc_prec}",  # perhaps you can make this 32 for CPU and 16 for IPU
            ],
        )
        if acc_type == "ipu":
            cfg["accelerator"]["ipu_config"].append("useIpuModel(True)")
            cfg["accelerator"]["ipu_inference_config"].append("useIpuModel(True)")

        run_training_finetuning_testing(cfg)
