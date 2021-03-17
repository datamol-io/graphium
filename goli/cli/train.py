import click

import fsspec
import omegaconf
from loguru import logger

import goli

from .main import main


@main.command(help="Train a Goli model.")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    required=True,
    help="Path to a Goli config file.",
)
def train(config_path):

    with fsspec.open(config_path) as f:
        config: omegaconf.DictConfig = omegaconf.OmegaConf.load(f)  # type: ignore

    logger.info("Load the data module.")

    if config.data.module_type == "DGLFromSmilesDataModule":
        dm = goli.data.DGLFromSmilesDataModule(**config.data.args)
    else:
        raise ValueError(f"The data module type '{config.data.module_type}' is not supported or not set.")

    logger.info("Build the model")

    logger.info("Setup training")

    logger.info("Start training")

    logger.info("Training is done")
