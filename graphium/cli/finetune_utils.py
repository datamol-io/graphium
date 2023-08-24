from typing import List, Optional

import fsspec
import torch
import tqdm
import typer
import yaml
from datamol.utils import fs
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import OmegaConf

from graphium.config._loader import load_accelerator, load_datamodule
from graphium.finetuning.fingerprinting import Fingerprinter
from graphium.utils import fs
from graphium.trainer.predictor import PredictorModule

from .main import app
from .train_finetune import run_training_finetuning

finetune_app = typer.Typer(help="Utility CLI for extra fine-tuning utilities.")
app.add_typer(finetune_app, name="finetune")


@finetune_app.command(name="admet")
def benchmark_tdc_admet_cli(
    overrides: List[str],
    name: Optional[List[str]] = None,
    inclusive_filter: bool = True,
):
    """
    Utility CLI to easily fine-tune a model on (a subset of) the benchmarks in the TDC ADMET group.
    A major limitation is that we cannot use all features of the Hydra CLI, such as multiruns.
    """
    try:
        from tdc.utils import retrieve_benchmark_names
    except ImportError:
        raise ImportError("TDC needs to be installed to use this CLI. Run `pip install PyTDC`.")

    # Get the benchmarks to run this for
    if len(name) == 0:
        name = retrieve_benchmark_names("admet_group")

    if not inclusive_filter:
        name = [n for n in retrieve_benchmark_names("admet_group") if n not in name]

    logger.info(f"Running fine-tuning for the following benchmarks: {name}")
    results = {}

    # Use the Compose API to construct the config
    for n in name:
        overrides += ["+finetuning=admet", f"constants.task={n}"]

        with initialize(version_base=None, config_path="../../expts/hydra-configs"):
            cfg = compose(
                config_name="main",
                overrides=overrides,
            )

        # Run the training loop
        ret = run_training_finetuning(cfg)
        ret = {k: v.item() for k, v in ret.items()}
        results[n] = ret

    # Save to the results_dir by default or to the Hydra output_dir if needed.
    # This distinction is needed, because Hydra's output_dir cannot be remote.
    save_dir = cfg["constants"].get("results_dir", HydraConfig.get()["runtime"]["output_dir"])
    fs.mkdir(save_dir, exist_ok=True)
    path = fs.join(save_dir, "results.yaml")
    logger.info(f"Saving results to {path}")

    with fsspec.open(path, "w") as f:
        yaml.dump(results, f)


@finetune_app.command(name="fp")
def get_fingerprints_from_model(
    fingerprint_layer_spec: List[str],
    pretrained_model: str,
    save_destination: str,
    overrides: Optional[List[str]] = typer.Option(None, "--override", "-o", help="Hydra overrides"),
):
    """Endpoint for getting fingerprints from a pretrained model."""

    if overrides is None:
        overrides = []

    with initialize(version_base=None, config_path="../../expts/hydra-configs"):
        cfg = compose(config_name="main", overrides=overrides)
        cfg = OmegaConf.to_container(cfg, resolve=True)

    ## == Instantiate all required objects from their respective configs ==

    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    # Data-module
    datamodule = load_datamodule(cfg, accelerator_type)
    datamodule.prepare_data()

    # The predict_dataloader() returns either predict or test, so we need to run both.
    datamodule.setup("test")
    datamodule.setup("predict")

    # Model
    model = PredictorModule.load_pretrained_models(
        pretrained_model,
        device=accelerator_type,
    ).model

    ## == Fingerprinter

    fps = []
    with Fingerprinter(network=model, fingerprint_spec=fingerprint_layer_spec) as fp:
        for batch in tqdm.tqdm(datamodule.predict_dataloader(), desc="Getting fingerprints"):
            # TODO (cwognum): This is a hack to make sure the features are float32 (instead of float16)
            #  I'm not sure why it is needed. Maybe we should try reuse the trainer.predict() endpoint from PTL?
            for k, v in batch["features"].items():
                if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                    batch["features"][k] = v.float()

            feats = fp.get_fingerprints_for_batch(batch["features"])
            fps.append(feats)

    fps = torch.cat(fps, dim=0)

    fs.mkdir(save_destination, exist_ok=True)
    path = fs.join(save_destination, "fingerprints.pt")

    logger.info(f"Saving fingerprints to {path}")
    torch.save(fps, path)
