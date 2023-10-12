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

@param_app.command(name="balance", help="Balance parameter count.")
def balance_parameter_count(overrides: List[str], target_param_count: int, max_rel_diff: float, rel_step: float, old_dim: int) -> None:
    with initialize(version_base=None, config_path="../../expts/hydra-configs"):
        cfg = compose(
            config_name="main",
            overrides=overrides,
        )

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Infer parameter count
    num_params = infer_parameter_count(overrides=overrides)

    # Get current hidden node and edge dim
    tmp_dim = cfg["constants"]["gnn_dim"]
    tmp_edge_dim = cfg["constants"]["gnn_edge_dim"]

    rel_diff = (num_params - target_param_count) / target_param_count

    # Balance parameter count when difference is too large
    if np.abs(rel_diff) > max_rel_diff:
        if rel_diff > 0:
            new_dim = int(tmp_dim * (1 - rel_step))
            new_edge_dim = int(tmp_edge_dim * (1 - rel_step))
        else:
            new_dim = int(tmp_dim * (1 + rel_step))
            new_edge_dim = int(tmp_edge_dim * (1 + rel_step))

        logger.info(f"Hidden node dim changed: {tmp_dim} -> {new_dim}.")
        logger.info(f"Hidden edge dim changed: {tmp_edge_dim} -> {new_edge_dim}.")

    else:
        logger.info(f"Hidden node dim unchanged: {tmp_dim}.")
        logger.info(f"Hidden edge dim unchanged: {tmp_edge_dim}.")
        print(tmp_dim, tmp_edge_dim, rel_step, "true")
        return
        
    # Reduce step size when overshooting
    if np.sign(old_dim - tmp_dim) != np.sign(tmp_dim - new_dim) and old_dim > 0:
        rel_step /= 2
        logger.info(f"Relative step changed: {2*rel_step} -> {rel_step}.")

    print(new_dim, new_edge_dim, rel_step, "false")
    
