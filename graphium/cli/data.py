import click

from loguru import logger

import graphium

from .main import main_cli


@main_cli.group(name="data", help="Graphium datasets.")
def data_cli():
    pass


@data_cli.command(name="download", help="Download a Graphium dataset.")
@click.option(
    "-n",
    "--name",
    type=str,
    required=True,
    help="Name of the graphium dataset to download.",
)
@click.option(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Where to download the Graphium dataset.",
)
@click.option(
    "--progress",
    type=bool,
    is_flag=True,
    default=False,
    required=False,
    help="Whether to extract the dataset if it's a zip file.",
)
def download(name, output, progress):
    args = {}
    args["name"] = name
    args["output_path"] = output
    args["extract_zip"] = True
    args["progress"] = progress

    logger.info(f"Download dataset '{name}' into {output}.")

    fpath = graphium.data.utils.download_graphium_dataset(**args)

    logger.info(f"Dataset available at {fpath}.")


@data_cli.command(name="list", help="List available Graphium dataset.")
def list():
    logger.info("Graphium datasets:")
    logger.info(graphium.data.utils.list_graphium_datasets())
