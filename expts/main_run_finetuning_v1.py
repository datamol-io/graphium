# General imports
import argparse
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import timeit
from loguru import logger
from datetime import datetime
import torch.nn as nn
from lightning.pytorch.utilities.model_summary import ModelSummary

from graphium.utils.mup import set_base_shapes
from graphium.finetuning import MolecularFinetuning

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
    load_yaml_config,
)
from graphium.utils.safe_run import SafeRun
from graphium.utils.command_line_utils import update_config, get_anchors_and_aliases


# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))

CONFIG_FILE = "expts/finetune_configs/config_toy_finetuning_v1.yaml"

os.chdir(MAIN_DIR)


def main(cfg: dict, run_name: str = "main", add_date_time: bool = True) -> None:
    st = timeit.default_timer()

    date_time_suffix = ""
    if add_date_time:
        date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    wandb.init(entity="multitask-gnn", project=cfg["constants"]["name"], config=cfg)

    # Initialize the accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg, accelerator_type)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dims=datamodule.in_dims,
    )

    datamodule.prepare_data()

    metrics = load_metrics(cfg)
    logger.info(metrics)

    predictor = load_predictor(
        cfg, model_class, model_kwargs, metrics, accelerator_type, datamodule.task_norms
    )


    ########################
    # Changes for finetuning

    # Load pretrained & replace in predictor
    pretrained_model = predictor.load_pretrained_models('dummy-pretrained-model').model     # make pretrained model part of config
    predictor.model = pretrained_model

    # Adapt pretrained model to new task
    # We need:
    #   - task head to finetune from
    #   - changes to the task head

    predictor.model.modify_architecture(cfg)

    # (Un)freezing will be handled by finetuning callback added to trainer

    predictor.model = set_base_shapes(predictor.model, base=None)       # how do we deal with muP for finetuning; it
    ########################


    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))


    trainer = load_trainer(cfg, run_name, accelerator_type, date_time_suffix)
    
    ########################
    # Changes for finetuning

    # Add the pl.BaseFinetuning callback to trainer
    
    trainer.callbacks.append(
        MolecularFinetuning(cfg)
    )
    ########################

    save_params_to_wandb(trainer.logger, cfg, predictor, datamodule)

    # Determine the max num nodes and edges in training and validation
    logger.info("About to set the max nodes etc.")
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])

    # Run the model training
    with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.fit(model=predictor, datamodule=datamodule)

    # Determine the max num nodes and edges in testing
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["test"])

    # Run the model testing
    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(model=predictor, datamodule=datamodule)  # , ckpt_path=ckpt_path)

    logger.info("--------------------------------------------")
    logger.info("total computation used", timeit.default_timer() - st)
    logger.info("--------------------------------------------")
    wandb.finish()

    return trainer.callback_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the config file", default=None)

    args, unknown_args = parser.parse_known_args()
    if args.config is not None:
        CONFIG_FILE = args.config
    cfg = load_yaml_config(CONFIG_FILE, MAIN_DIR, unknown_args)

    main(cfg)
