from typing import Optional

import click

from mlem.cli.main import mlem_command, option_repo, option_rev


@mlem_command("get")
@click.argument("uri")
@click.option("-o", "--out", help="Path to store the downloaded object.")
@option_repo
@option_rev
def get(uri: str, out: str, repo: Optional[str], rev: Optional[str]):
    """Download MLEM object from {uri} and save it to {out}."""
    from mlem.api.commands import get

    click.echo(f"Downloading {uri} to {out}")
    get(uri, out, repo=repo, rev=rev)
