from typing import Optional

import click

from mlem.cli.main import cli
from mlem.core.errors import InvalidArgumentError
from mlem.core.meta_io import MLEM_DIR


@cli.command("link")
@click.argument("source")
@click.option(
    "--repo", default=None, help="Repo in which {source} can be found."
)
@click.option("--rev", default=None, help="Repo revision to use.")
@click.argument("target")
@click.option(
    "--mlem-root",
    default=None,
    help="Save link to mlem dir found in {mlem_root} path.",
)
@click.option(
    "--no-mlem-root",
    "-o",
    default=False,
    is_flag=True,
    help=f"Save link not in {MLEM_DIR}, but as a plain file.",
)
@click.option(
    "--follow-links/--no-follow-links",
    default=True,
    help="If True, first follow links while reading {source} before creating this link.",
)
@click.option(
    "--check-extension/--no-check-extension",
    default=True,
    help="If True and --no-mlem-root specified, check that {target} endswith MLEM extension.",
)
@click.option(
    "--absolute/--relative",
    "--abs/--rel",
    default=False,
    help="Which path to linked object to specify: absolute or relative.",
)
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
                "--mlem-root and --no-mlem-root are mitually exclusive."
            )
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
