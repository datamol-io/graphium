from typing import Any, List, Dict

from loguru import logger

from omegaconf import OmegaConf

import wandb

from graphium.fingerprinting.data import FingerprintDatamodule

import typer
from hydra import initialize, compose

from graphium.cli.main import app

fp_app = typer.Typer(help="Automated fingerprinting from pretrained models.")
app.add_typer(fp_app, name="fps")

@fp_app.command(name="create", help="Create fingerprints for pretrained model.")
def smiles_to_fps(cfg_name: str, overrides: List[str]) -> Dict[str, Any]:
    with initialize(version_base=None, config_path="../../expts/hydra-configs/fingerprinting"):
        cfg = compose(
            config_name=cfg_name,
            overrides=overrides,
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(
        name="Fingerprinting",
        entity="valencelabs",
        project="graphium-fingerprints",
    )

    pretrained_models = cfg.get("pretrained")

    # Allow alternative definition of `pretrained_models` with the single model specifier and desired layers
    if "layers" in pretrained_models.keys():
        assert "model" in pretrained_models.keys(), "this workflow allows easier definition of fingerprinting sweeps"
        model, layers = pretrained_models.pop("model"), pretrained_models.pop("layers")
        pretrained_models[model] = layers
        
    data_kwargs = cfg.get("datamodule")
    
    datamodule = FingerprintDatamodule(
        pretrained_models=pretrained_models,
        **data_kwargs,
    )

    datamodule.prepare_data()

    logger.info(f"Fingerprints saved in {datamodule.fps_cache_dir}/fps.pt.")
    wandb.run.finish()


if __name__ == "__main__":
    smiles_to_fps(cfg_name="tdc", overrides=[])