from typing import Optional

import click

from mlem.cli.main import mlem_command
from mlem.core.meta_io import MLEM_DIR


@mlem_command("link")
@click.argument("source")
@click.option(
    "--repo", default=None, help="Repo in which {source} can be found."
)
@click.option("--rev", default=None, help="Repo revision to use.")
@click.option(
    "--source-mlem-root",
    "--sr",
    default=None,
    help="Load source from mlem repo found in {mlem_root} path.",
)
@click.argument("target")
@click.option(
    "--target-mlem-root",
    "--tr",
    default=None,
    help="Save link to mlem dir found in {mlem_root} path.",
)
@click.option(
    "--external",
    "-e",
    default=False,
    is_flag=True,
    help=f"Save link not in {MLEM_DIR}, but as a plain file.",
)
@click.option(
    "--follow-links/--no-follow-links",
    "--f/--nf",
    default=True,
    help="If True, first follow links while reading {source} before creating this link.",
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
    source_mlem_root: Optional[str],
    target: str,
    target_mlem_root: Optional[str],
    external: bool,
    follow_links: bool,
    absolute: bool,
):
    """Create link for {source} MLEM object and place it in {target}."""
    from mlem.api.commands import link

    link(
        source=source,
        repo=repo,
        rev=rev,
        source_mlem_root=source_mlem_root,
        target=target,
        target_mlem_root=target_mlem_root,
        follow_links=follow_links,
        external=external,
        absolute=absolute,
    )
