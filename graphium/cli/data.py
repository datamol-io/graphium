import typer
import graphium

from loguru import logger

from .main import app


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
