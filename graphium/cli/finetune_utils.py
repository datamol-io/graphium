from typing import List, Optional

import fsspec
import typer
import yaml
from datamol.utils import fs
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from loguru import logger

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
