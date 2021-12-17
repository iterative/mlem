from typing import Optional

import click

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_link,
    option_repo,
    option_rev,
    option_target_repo,
)


@mlem_command("import")
@click.argument("uri")
@click.argument("target", help="Path to store the imported object.")
@click.option(
    "--move/--no-move",
    default=True,
    is_flag=True,
    help="Whether to move files to obj dir",
)
@click.option("--type", "type_", default=None)
@option_repo
@option_rev
@option_link
@option_external
@option_target_repo
def import_path(
    uri: str,
    repo: Optional[str],
    rev: Optional[str],
    target: str,
    target_repo: Optional[str],
    copy_data: bool,
    type_: str,
    link: bool,
    external: bool,
):
    """Make MLEM model (or dataset) out of object found at {uri} and save it to {out}."""
    from mlem.api.commands import import_object

    click.echo(f"Importing {uri} to {target}")
    import_object(
        uri,
        repo=repo,
        rev=rev,
        target=target,
        target_repo=target_repo,
        copy_data=copy_data,
        type_=type_,
        external=external,
        link=link,
    )
