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


@mlem_command("clone")
@click.argument("uri")
@click.option("-t", "--target", help="Path to store the downloaded object.")
@option_repo
@option_rev
@option_target_repo
@option_link
@option_external
def clone(
    uri: str,
    target: str,
    repo: Optional[str],
    rev: Optional[str],
    target_repo: Optional[str],
    external: Optional[bool],
    link: Optional[bool],
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
