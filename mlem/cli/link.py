from typing import Optional

import click

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_rev,
    option_target_repo,
)


@mlem_command("link")
@click.argument("source")
@option_rev
@click.option(
    "--source-repo",
    "--sr",
    default=None,
    help="Load source from mlem repo found in {source_repo} path.",
)
@click.argument("target")
@option_target_repo
@option_external
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
    source_repo: Optional[str],
    rev: Optional[str],
    target: str,
    target_repo: Optional[str],
    external: bool,
    follow_links: bool,
    absolute: bool,
):
    """Create link for {source} MLEM object and place it in {target}."""
    from mlem.api.commands import link

    link(
        source=source,
        source_repo=source_repo,
        rev=rev,
        target=target,
        target_repo=target_repo,
        follow_links=follow_links,
        external=external,
        absolute=absolute,
    )
