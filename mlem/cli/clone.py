from typing import Optional

import click

from mlem.cli.main import mlem_command, option_repo, option_rev


@mlem_command("clone")
@click.argument("uri")
@click.option("-t", "--target", help="Path to store the downloaded object.")
@option_repo
@option_rev
def clone(uri: str, target: str, repo: Optional[str], rev: Optional[str]):
    """Download MLEM object from {uri} and save it to {out}."""
    from mlem.api.commands import clone

    click.echo(f"Downloading {uri} to {target}")
    clone(uri, target, repo=repo, rev=rev, target_repo=target_repo, external=external, link=link)
