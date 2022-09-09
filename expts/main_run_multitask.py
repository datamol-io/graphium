# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import timeit
from loguru import logger

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_metrics, load_architecture, load_predictor, load_trainer
from goli.utils.safe_run import SafeRun


# WandB
import wandb

# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
# CONFIG_FILE = "expts/configs/config_micro_ZINC_mtl_test_3_tasks_pyg.yaml"
CONFIG_FILE = "expts/configs/config_ipu_allsizes.yaml"
# CONFIG_FILE = "expts/configs/config_ipu_reproduce.yaml"
os.chdir(MAIN_DIR)


def main(cfg: DictConfig, run_name="main") -> None:
    st = timeit.default_timer()

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
        in_dim_edges=datamodule.num_edge_feats,
    )

    metrics = load_metrics(cfg)
    logger.info(metrics)

    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)

    logger.info(predictor.model)
    logger.info(predictor.summarize(max_depth=4))

    trainer = load_trainer(cfg, run_name)

    datamodule.prepare_data()
    # Run the model training
    with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.fit(model=predictor, datamodule=datamodule)

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
