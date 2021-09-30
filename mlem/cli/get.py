import click

from mlem.cli.main import cli


@cli.command("get")
@click.argument("uri")
@click.option("-o", "--out")
def get(uri: str, out: str):
    """Download MLEM object from {uri} and save it to {out}"""
    from mlem.api.commands import get

    click.echo(f"Downloading {uri} to {out}")
    get(uri, out)
