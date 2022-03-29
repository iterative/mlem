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


@mlem_command("clone")
def clone(
    uri: str,
    target: str = Option(
        None, "-t", "--target", help="Path to store the downloaded object."
    ),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    target_repo: Optional[str] = option_target_repo,
    external: Optional[bool] = option_external,
    link: Optional[bool] = option_link,
):
    """Download MLEM object from {uri} and save it to {out}."""
    from mlem.api.commands import clone

    click.echo(f"Downloading {uri} to {target}")
    clone(
        uri,
        target,
        repo=repo,
        rev=rev,
        target_repo=target_repo,
        external=external,
        link=link,
    )
