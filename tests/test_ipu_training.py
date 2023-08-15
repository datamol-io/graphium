import hydra
import wandb
import timeit

from omegaconf import DictConfig, OmegaConf
from loguru import logger
from datetime import datetime
from lightning.pytorch.utilities.model_summary import ModelSummary

from graphium.config._loader import (
    load_datamodule,
    load_metrics,
    load_architecture,
    load_predictor,
    load_trainer,
    load_accelerator,
    save_params_to_wandb,
)
from graphium.finetuning import modify_cfg_for_finetuning, GraphFinetuning
from graphium.utils.safe_run import SafeRun

from graphium.cli.train_finetune import run_training_finetuning
import pytest

FINETUNING_CONFIG_KEY = "finetuning"


@pytest.mark.parametrize("acc_type", ["cpu", "ipu"])
@pytest.mark.ipu
def test_cli(acc_type) -> None:
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
            ],
        )
        if acc_type == "ipu":
            cfg["accelerator"]["ipu_config"].append("useIpuModel(True)")
            cfg["accelerator"]["ipu_inference_config"].append("useIpuModel(True)")

        run_training_finetuning(cfg)
