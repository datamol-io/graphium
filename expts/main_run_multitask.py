# General imports
import os
from os.path import dirname, abspath
from omegaconf import DictConfig, OmegaConf
import timeit
from loguru import logger
from datetime import datetime
from lightning.pytorch.utilities.model_summary import ModelSummary

# Current project imports
import graphium
from graphium.config._loader import (
    load_datamodule,
    load_metrics,
    load_architecture,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
    load_accelerator,
)
from graphium.utils.safe_run import SafeRun

import hydra

# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
os.chdir(MAIN_DIR)


@hydra.main(version_base=None, config_path="hydra-configs", config_name="main")
def main(cfg: DictConfig) -> None:
    raise DeprecationWarning(
        "This script is deprecated. Use `python graphium/cli/train_finetune.py` (or `graphium-train`) instead!"
    )


if __name__ == "__main__":
    main()
