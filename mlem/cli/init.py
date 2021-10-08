import click

from mlem.cli.main import mlem_command


@mlem_command("init")
@click.argument("path", default=".")
def init(path: str):
    """Create .mlem folder in {path}"""
    from mlem.api.commands import init

    init(path)
