import click


@click.group()
@click.version_option()
def main():
    pass


@main.command()
def hello():
    click.echo("Hello")


if __name__ == "__main__":
    main()
