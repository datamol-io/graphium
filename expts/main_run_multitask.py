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
import goli
from goli.config._loader import (
    load_datamodule,
    load_metrics,
    load_architecture,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
    load_accelerator,
)
from goli.utils.safe_run import SafeRun


# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))

# CONFIG_FILE = "expts/configs/config_mpnn_10M_b3lyp.yaml"
# CONFIG_FILE = "expts/configs/config_mpnn_10M_pcqm4m.yaml"
CONFIG_FILE = "expts/neurips2023_configs/config_small_mpnn.yaml"
# CONFIG_FILE = "expts/neurips2023_configs/config_large_mpnn.yaml"
os.chdir(MAIN_DIR)

class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super(ConfigDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError("No such attribute: " + key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError("No such attribute: " + key)

    def to_dict(self):
        return {key: self[key].to_dict() if isinstance(self[key], ConfigDict) else self[key]
                for key in self}

def main(cfg: DictConfig, run_name: str = "main", add_date_time: bool = True) -> None:
    st = timeit.default_timer()

    date_time_suffix = ""
    if add_date_time:
        date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    cfg = deepcopy(cfg)
    print(cfg)
    wandb.config = cfg
    cfg = wandb.config

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


def update_config(cfg: ConfigDict, unknown: list):
    """
    Update the configuration dictionary with command line arguments.
    """
    for arg in unknown:
        if arg.startswith("--"):
            key, value = arg[2:].split('=')
            keys = key.split('.')
            temp_cfg = cfg
            for k in keys[:-1]:
                temp_cfg = temp_cfg[k]
            temp_cfg[keys[-1]] = type(temp_cfg[keys[-1]])(value) if keys[-1] in temp_cfg else value
    return cfg



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')

    args, unknown = parser.parse_known_args()

    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
        cfg = ConfigDict(cfg)
        cfg = update_config(cfg, unknown)
    main(cfg.to_dict())

