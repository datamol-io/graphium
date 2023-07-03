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
from pytorch_lightning.utilities.model_summary import ModelSummary

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


# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))

# CONFIG_FILE = "expts/configs/config_mpnn_10M_b3lyp.yaml"
# CONFIG_FILE = "expts/configs/config_mpnn_10M_pcqm4m.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_debug.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_large_mpnn.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_large_gcn.yaml"
CONFIG_FILE = "expts/neurips2023_configs/debug/config_large_gcn_debug.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_large_gin.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_large_gine.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_small_gcn.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_large_gcn.yaml"
# CONFIG_FILE = "exptas/neurips2023_configs/config_small_gin.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_small_gine.yaml"
os.chdir(MAIN_DIR)


def main(cfg: DictConfig, run_name: str = "main", add_date_time: bool = True) -> None:
    st = timeit.default_timer()

    date_time_suffix = ""
    if add_date_time:
        date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    cfg = deepcopy(cfg)
    wandb.init(project=cfg["constants"]["name"], config=cfg)

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
    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    trainer = load_trainer(cfg, run_name, accelerator_type, date_time_suffix)
    save_params_to_wandb(trainer.logger, cfg, predictor, datamodule)

    # Determine the max num nodes and edges in training and validation
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["val"])

    # Run the model validation
    with SafeRun(name="VALIDATING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.validate(
            model=predictor,
            ckpt_path=f'{cfg["trainer"]["model_checkpoint"]["dirpath"]}{cfg["trainer"]["seed"]}/{cfg["trainer"]["model_checkpoint"]["filename"]}.ckpt',
            datamodule=datamodule,
        )

    # Run the model testing
    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(
            model=predictor,
            ckpt_path=f'{cfg["trainer"]["model_checkpoint"]["dirpath"]}{cfg["trainer"]["seed"]}/{cfg["trainer"]["model_checkpoint"]["filename"]}.ckpt',
            datamodule=datamodule,
        )

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
