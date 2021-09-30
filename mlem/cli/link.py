from typing import Optional

import click

from mlem.cli.main import cli
from mlem.core.errors import InvalidArgumentError
from mlem.core.meta_io import MLEM_DIR


@cli.command("link")
@click.argument("source")
@click.option("--repo", default=None)
@click.option("--rev", default=None)
@click.argument("target")
@click.option("--mlem-root", default=None)
@click.option(
    "--no-mlem-root",
    "-o",
    default=False,
    is_flag=True,
    help=f"Save link not in {MLEM_DIR}, but as a plain file",
)
@click.option("--follow-links/--no-follow-links", default=True)
@click.option("--check-extension/--no-check-extension", default=True)
@click.option("--absolute/--relative", "--abs/--rel", default=False)
def link(
    source: str,
    repo: Optional[str],
    rev: Optional[str],
    target: str,
    mlem_root: Optional[str],
    no_mlem_root: bool,
    follow_links: bool,
    check_extension: bool,
    absolute: bool,
):
    """Create link for {source} MLEM object and place it in {target}."""
    from mlem.api.commands import link

    if no_mlem_root:
        if mlem_root is not None:
            raise InvalidArgumentError(
                "--mlem-root and --no-mlem-root are mitually exclusive"
            )
        else:
            mlem_root = None
    elif mlem_root is None:
        mlem_root = "."

    link(
        source=source,
        repo=repo,
        rev=rev,
        target=target,
        mlem_root=mlem_root,
        follow_links=follow_links,
        check_extension=check_extension,
        absolute=absolute,
    )
