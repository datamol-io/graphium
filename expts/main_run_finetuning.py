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
from graphium.utils.safe_run import SafeRun

import hydra

# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(graphium.__file__)))
os.chdir(MAIN_DIR)


@hydra.main(version_base=None, config_path="hydra-configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg, resolve=True)

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

    ########################
    # Changes for finetuning

    # Load pretrained & replace in predictor
    pretrained_model = predictor.load_pretrained_models(
        cfg["finetuning"]["pretrained_model"]
    ).model  # make pretrained model part of config  # use latest or best available checkpoint

    # Adapt pretrained model to new task
    # We need to overwrite shared weights with pretrained

    predictor.model.overwrite_with_pretrained(cfg, pretrained_model)

    # (Un)freezing will be handled by finetuning callback added to trainer

    predictor.model = set_base_shapes(predictor.model, base=None)  # how do we deal with muP for finetuning
    ########################

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    trainer = load_trainer(cfg, run_name, accelerator_type, date_time_suffix)

    ########################
    # Changes for finetuning

    # Add the pl.BaseFinetuning callback to trainer

    trainer.callbacks.append(GraphFinetuning(cfg))
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
