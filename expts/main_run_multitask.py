# General imports
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
import goli
from goli.config._loader import (
    load_datamodule,
    load_metrics,
    load_architecture,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
)
from goli.utils.safe_run import SafeRun


# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))

# CONFIG_FILE = "expts/configs/config_ipu_pcqm4m.yaml"
# CONFIG_FILE = "expts/configs/config_ipu_qm9.yaml"
# CONFIG_FILE = "expts/configs/config_gpu_qm9.yaml"
# CONFIG_FILE = "expts/configs/config_qm9_gpspp.yaml"
CONFIG_FILE = "expts/configs/config_pcqmv2_mpnn.yaml"

# CONFIG_FILE = "expts/configs/config_ipu_reproduce.yaml"
os.chdir(MAIN_DIR)


def main(cfg: DictConfig, run_name: str = "main", add_date_time: bool = True) -> None:
    st = timeit.default_timer()

    if add_date_time:
        run_name += "_" + datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dims=datamodule.in_dims,
    )

    metrics = load_metrics(cfg)
    logger.info(metrics)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    trainer = load_trainer(cfg, run_name)
    save_params_to_wandb(trainer.logger, cfg, predictor, datamodule)

    datamodule.prepare_data()

    # Determine the max num nodes and edges in training and validation
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
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
