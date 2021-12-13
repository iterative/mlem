from typing import Optional

import click

from mlem.cli.main import mlem_command, option_repo, option_rev


@mlem_command("import")
@click.argument("uri")
@click.argument("out", help="Path to store the imported object.")
@click.option(
    "--move/--no-move",
    default=True,
    is_flag=True,
    help="Whether to move files to obj dir",
)
@click.option("--type", "type_", default=None)
@option_repo
@option_rev
def import_path(
    uri: str,
    out: str,
    repo: Optional[str],
    rev: Optional[str],
    move: bool,
    type_: str,
):
    """Make MLEM model (or dataset) out of object found at {uri} and save it to {out}."""
    from mlem.api.commands import import_path

    click.echo(f"Importing {uri} to {out}")
    import_path(uri, out, repo=repo, rev=rev, move=move, type_=type_)
