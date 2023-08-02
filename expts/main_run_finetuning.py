from typing import Union, List
from copy import deepcopy

# General imports
import os
from os.path import dirname, abspath
from omegaconf import DictConfig, OmegaConf
import timeit
from loguru import logger
from datetime import datetime
from lightning.pytorch.utilities.model_summary import ModelSummary

from graphium.utils.mup import set_base_shapes
from graphium.finetuning import modify_cfg_for_finetuning, GraphFinetuning

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
from graphium.trainer import PredictorModule

from graphium.utils.safe_run import SafeRun

import hydra

# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
os.chdir(MAIN_DIR)


def filter_cfg_based_on_benchmark_name(config, names: Union[List[str], str]):
    """
    Filter a base config for the full TDC ADMET benchmarking group to only
    have settings related to a subset of the endpoints
    """

    if config["datamodule"]["module_type"] != "ADMETBenchmarkDataModule":
        raise ValueError("You can only use this method for the `ADMETBenchmarkDataModule`")

    if isinstance(names, str):
        names = [names]

    def _filter(d):
        return {k: v for k, v in d.items() if k in names}

    cfg = deepcopy(config)

    # Update the datamodule arguments
    cfg["datamodule"]["args"]["tdc_benchmark_names"] = names

    # Filter the relevant config sections
    cfg["architecture"]["task_heads"] = _filter(cfg["architecture"]["task_heads"])
    cfg["predictor"]["metrics_on_progress_bar"] = _filter(cfg["predictor"]["metrics_on_progress_bar"])
    cfg["predictor"]["loss_fun"] = _filter(cfg["predictor"]["loss_fun"])
    cfg["metrics"] = _filter(cfg["metrics"])

    return cfg


@hydra.main(version_base=None, config_path="hydra-configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    names = cfg["datamodule"]["args"]["tdc_benchmark_names"]
    # cfg = filter_cfg_based_on_benchmark_name(cfg, names)

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # PredictorModule.load_from_checkpoint("tests/dummy-pretrained-model.ckpt")

    cfg = modify_cfg_for_finetuning(cfg)

    run_name: str = "main"
    add_date_time: bool = True

    st = timeit.default_timer()

    date_time_suffix = ""
    if add_date_time:
        date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    wandb.init(entity=cfg["constants"]["entity"], project=cfg["constants"]["name"], config=cfg)

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

    ########################
    # Changes for finetuning

    # Add the pl.BaseFinetuning callback to trainer
    cfg_arch, finetuning_training_kwargs = cfg["architecture"], cfg["finetuning"]["training_kwargs"]
    trainer.callbacks.append(GraphFinetuning(cfg_arch, **finetuning_training_kwargs))
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
    main()
