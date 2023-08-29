import timeit
from typing import List
from omegaconf import OmegaConf
import typer
import graphium

from loguru import logger
from hydra import initialize, compose

from .main import app
from graphium.config._loader import load_datamodule


data_app = typer.Typer(help="Graphium datasets.")
app.add_typer(data_app, name="data")


@data_app.command(name="download", help="Download a Graphium dataset.")
def download(name: str, output: str, progress: bool = True):
    args = {}
    args["name"] = name
    args["output_path"] = output
    args["extract_zip"] = True
    args["progress"] = progress

    logger.info(f"Download dataset '{name}' into {output}.")

    fpath = graphium.data.utils.download_graphium_dataset(**args)

    logger.info(f"Dataset available at {fpath}.")


@data_app.command(name="list", help="List available Graphium dataset.")
def list():
    logger.info("Graphium datasets:")
    logger.info(graphium.data.utils.list_graphium_datasets())


@data_app.command(name="prepare", help="Prepare a Graphium dataset.")
def prepare_data(overrides: List[str]) -> None:
    with initialize(version_base=None, config_path="../../expts/hydra-configs"):
        cfg = compose(
            config_name="main",
            overrides=overrides,
        )
    cfg = OmegaConf.to_container(cfg, resolve=True)
    st = timeit.default_timer()

    # Checking that `processed_graph_data_path` is provided
    path = cfg["datamodule"]["args"].get("processed_graph_data_path", None)
    if path is None:
        raise ValueError(
            "Please provide `datamodule.args.processed_graph_data_path` to specify the caching dir."
        )
    logger.info(f"The caching dir is set to '{path}'")

    # Data-module
    datamodule = load_datamodule(cfg, "cpu")
    datamodule.prepare_data()

    logger.info(f"Data preparation took {timeit.default_timer() - st:.2f} seconds.")
