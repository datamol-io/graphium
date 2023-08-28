import hydra
import wandb
import timeit

from omegaconf import DictConfig, OmegaConf
from loguru import logger
from datetime import datetime
from lightning.pytorch.utilities.model_summary import ModelSummary
from graphium.trainer.predictor import PredictorModule

from graphium.config._loader import (
    load_datamodule,
    get_checkpoint_path,
    load_trainer,
    load_accelerator,
)
from graphium.utils.safe_run import SafeRun


@hydra.main(version_base=None, config_path="../../expts/hydra-configs", config_name="main")
def cli(cfg: DictConfig) -> None:
    """
    CLI endpoint for running test step on model checkpoints.
    """
    run_testing(cfg)


def run_testing(cfg: DictConfig) -> None:
    """
    The main (pre-)training and fine-tuning loop.
    """

    cfg = OmegaConf.to_container(cfg, resolve=True)

    st = timeit.default_timer()

    wandb_cfg = cfg["constants"].get("wandb")
    if wandb_cfg is not None:
        wandb.init(
            entity=wandb_cfg["entity"],
            project=wandb_cfg["project"],
            config=cfg,
        )

    ## == Instantiate all required objects from their respective configs ==
    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    ## Data-module
    datamodule = load_datamodule(cfg, accelerator_type)

    ## Load Predictor
    predictor = PredictorModule.load_from_checkpoint(
        checkpoint_path=get_checkpoint_path(cfg), map_location=cfg["accelerator"]["type"]
    )

    ## Load Trainer
    date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    trainer = load_trainer(cfg, accelerator_type, date_time_suffix)

    # Determine the max num nodes and edges in testing
    datamodule.setup(stage="test")

    max_nodes = datamodule.get_max_num_nodes_datamodule(stages=["test"])
    max_edges = datamodule.get_max_num_edges_datamodule(stages=["test"])

    predictor.model.set_max_num_nodes_edges_per_graph(max_nodes, max_edges)

    # Run the model testing
    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(model=predictor, datamodule=datamodule)

    logger.info("-" * 50)
    logger.info("Total compute time:", timeit.default_timer() - st)
    logger.info("-" * 50)

    if wandb_cfg is not None:
        wandb.finish()

    return trainer.callback_metrics


if __name__ == "__main__":
    cli()
