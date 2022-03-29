from typing import Optional

import click
from typer import Option

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_link,
    option_repo,
    option_rev,
    option_target_repo,
)


@mlem_command("import")
def import_path(
    uri: str,
    target: str,
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    target_repo: Optional[str] = option_target_repo,
    copy: bool = Option(
        True,
        help="Whether to create a copy of file in target location or just link existing file",
    ),
    type_: Optional[str] = Option(None, "--type"),
    link: bool = option_link,
    external: bool = option_external,
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
        copy_data=copy,
        type_=type_,
        external=external,
        link=link,
    )
