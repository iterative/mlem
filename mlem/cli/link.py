from typing import Optional

from typer import Option

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_rev,
    option_target_repo,
)


@mlem_command("link")
def link(
    source: str,
    target: str,
    source_repo: Optional[str] = Option(
        None,
        "--source-repo",
        "--sr",
        help="Load source from mlem repo found in {source_repo} path.",
    ),
    rev: Optional[str] = option_rev,
    target_repo: Optional[str] = option_target_repo,
    external: bool = option_external,
    follow_links: bool = Option(
        True,
        "--follow-links/--no-follow-links",
        "--f/--nf",
        help="If True, first follow links while reading {source} before creating this link.",
    ),
    absolute: bool = Option(
        False,
        "--absolute/--relative",
        "--abs/--rel",
        help="Which path to linked object to specify: absolute or relative.",
    ),
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
