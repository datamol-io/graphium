import timeit
from typing import List
from omegaconf import DictConfig, OmegaConf
import typer

import numpy as np

from loguru import logger
from hydra import initialize, compose

from .main import app
from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_datamodule,
)

from graphium.trainer.predictor_options import ModelOptions


param_app = typer.Typer(help="Parameter counts.")
app.add_typer(param_app, name="params")


@param_app.command(name="infer", help="Infer parameter count.")
def infer_parameter_count(overrides: List[str] = []) -> int:
    with initialize(version_base=None, config_path="../../expts/hydra-configs"):
        cfg = compose(
            config_name="main",
            overrides=overrides,
        )

    cfg = OmegaConf.to_container(cfg, resolve=True)

    ## Accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    ## Datamodule
    datamodule = load_datamodule(cfg, accelerator_type)

    ## Architecture
    model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)
    model_options = ModelOptions(
        model_class=model_class,
        model_kwargs=model_kwargs,
    )
    model = model_options.model_class(**model_options.model_kwargs)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Number of parameters: {num_params}.")

    return num_params
