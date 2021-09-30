import click

from mlem.cli.main import cli


@cli.command("init")
@click.argument("path", default=".")
def init(path: str):
    """Create .mlem folder in {path}"""
    from mlem.api.commands import init

    init(path)
