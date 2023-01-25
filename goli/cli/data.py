import click

from loguru import logger

import goli

from .main import main_cli


@main_cli.group(name="data", help="Goli datasets.")
def data_cli():
    pass


@data_cli.command(name="download", help="Download a Goli dataset.")
@click.option(
    "-n",
    "--name",
    type=str,
    required=True,
    help="Name of the goli dataset to download.",
)
@click.option(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Where to download the Goli dataset.",
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

    fpath = goli.data.utils.download_goli_dataset(**args)

    logger.info(f"Dataset available at {fpath}.")


@data_cli.command(name="list", help="List available Goli dataset.")
def list():
    logger.info("Goli datasets:")
    logger.info(goli.data.utils.list_goli_datasets())
