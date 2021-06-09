import click


@click.group()
@click.version_option()
def main_cli():
    pass


if __name__ == "__main__":
    main_cli()
