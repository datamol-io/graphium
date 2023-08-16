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


FINETUNING_CONFIG_KEY = "finetuning"


@hydra.main(version_base=None, config_path="../../expts/hydra-configs", config_name="main")
def cli(cfg: DictConfig) -> None:
    """
    The main CLI endpoint for training and fine-tuning Graphium models.
    """
    run_training_finetuning(cfg)


def run_training_finetuning(cfg: DictConfig) -> None:
    """
    The main (pre-)training and fine-tuning loop.
    """

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Modify the config for finetuning
    if FINETUNING_CONFIG_KEY in cfg:
        cfg = modify_cfg_for_finetuning(cfg)

    st = timeit.default_timer()

    # Disable wandb if the user is not logged in.
    wandb_cfg = cfg["constants"].get("wandb")
    if wandb.login() is False:
        logger.info(
            "Not logged in to wandb - disabling wandb logging.\n"
            + "To enable wandb, run `wandb login` from the command line."
        )
        wandb.init(mode="disabled")
    elif wandb_cfg is not None:
        wandb.init(config=cfg, **wandb_cfg)

    ## == Instantiate all required objects from their respective configs ==
    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    ## Data-module
    datamodule = load_datamodule(cfg, accelerator_type)

    ## Architecture
    model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)

    datamodule.prepare_data()

    ## Metrics
    metrics = load_metrics(cfg)

    ## Predictor
    predictor = load_predictor(
        config=cfg,
        model_class=model_class,
        model_kwargs=model_kwargs,
        metrics=metrics,
        task_levels=datamodule.get_task_levels(),
        accelerator_type=accelerator_type,
        featurization=datamodule.featurization,
        task_norms=datamodule.task_norms,
    )

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    ## Trainer
    date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    trainer = load_trainer(cfg, accelerator_type, date_time_suffix)

    # Add the fine-tuning callback to trainer
    if FINETUNING_CONFIG_KEY in cfg:
        finetuning_training_kwargs = cfg["finetuning"]["training_kwargs"]
        trainer.callbacks.append(GraphFinetuning(**finetuning_training_kwargs))

    if wandb_cfg is not None:
        save_params_to_wandb(trainer.logger, cfg, predictor, datamodule)

    # Determine the max num nodes and edges in training and validation
    logger.info("Computing the maximum number of nodes and edges per graph")
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])

    # Run the model training
    with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.fit(model=predictor, datamodule=datamodule)

    # Determine the max num nodes and edges in testing
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["test"])

    # Run the model testing
    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(model=predictor, datamodule=datamodule)  # , ckpt_path=ckpt_path)

    logger.info("-" * 50)
    logger.info("Total compute time:", timeit.default_timer() - st)
    logger.info("-" * 50)

    if wandb_cfg is not None:
        wandb.finish()

    return trainer.callback_metrics


if __name__ == "__main__":
    cli()
