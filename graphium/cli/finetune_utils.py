import yaml
import click
import fsspec

from loguru import logger
from hydra import compose, initialize
from datamol.utils import fs

from .main import main_cli
from .train_finetune import run_training_finetuning


@main_cli.group(name="finetune", help="Utility CLI for extra fine-tuning utilities.")
def finetune_cli():
    pass


@finetune_cli.command(name="admet")
@click.argument("save_dir")
@click.option("--wandb/--no-wandb", default=True, help="Whether to log to Weights & Biases.")
@click.option(
    "--name",
    "-n",
    multiple=True,
    help="One or multiple benchmarks to filter on. See also --inclusive-filter/--exclusive-filter.",
)
@click.option(
    "--inclusive-filter/--exclusive-filter",
    default=True,
    help="Whether to include or exclude the benchmarks specified by `--name`.",
)
def benchmark_tdc_admet_cli(save_dir, wandb, name, inclusive_filter):
    """
    Utility CLI to easily fine-tune a model on (a subset of) the benchmarks in the TDC ADMET group.
    The results are saved to the SAVE_DIR.
    """

    try:
        from tdc.utils import retrieve_benchmark_names
    except ImportError:
        raise ImportError("TDC needs to be installed to use this CLI. Run `pip install PyTDC`.")

    # Get the benchmarks to run this for
    if name is None:
        name = retrieve_benchmark_names("admet_group")
    elif not inclusive_filter:
        name = [n for n in name if n not in retrieve_benchmark_names("admet_group")]

    results = {}

    # Use the Compose API to construct the config
    for n in name:
        overrides = [
            "+finetuning=admet",
            f"finetuning.task={n}",
            f"finetuning.finetuning_head.task={n}",
        ]

        if not wandb:
            overrides.append("~constants.wandb")

        with initialize(version_base=None, config_path="../../expts/hydra-configs"):
            cfg = compose(
                config_name="main",
                overrides=overrides,
            )

        # Run the training loop
        ret = run_training_finetuning(cfg)
        ret = {k: v.item() for k, v in ret.items()}
        results[n] = ret

    fs.mkdir(save_dir, exist_ok=True)
    path = fs.join(save_dir, "results.yaml")
    logger.info(f"Saving results to {path}")

    with fsspec.open(path, "w") as f:
        yaml.dump(results, f)
