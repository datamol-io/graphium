import click
from hydra import compose, initialize

from .main import main_cli
from .hydra import run_training_finetuning


@main_cli.group(name="finetune", help="Utility CLI for easy fine-tuning.")
def finetune_cli():
    pass


@finetune_cli.command(name="admet")
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
def benchmark_tdc_admet_cli(wandb, name, inclusive_filter):
    """
    Utility CLI to easily fine-tune a model on (a subset of) the benchmarks in the TDC ADMET group.
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
        results = run_training_finetuning(cfg)
        print(results)

    print("Done!")
