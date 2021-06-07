import click

from loguru import logger

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
def download(name, output):

    logger.info(name)
    logger.info(output)


@data_cli.command(name="list", help="List available Goli dataset.")
def list():

    logger.info("hello")
