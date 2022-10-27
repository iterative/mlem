from typing import Optional

from typer import Argument, Option

from mlem.cli.main import (
    PATH_METAVAR,
    mlem_command,
    option_rev,
    option_target_project,
)


@mlem_command("link", section="object")
def link(
    source: str = Argument(
        ..., help="URI of the MLEM object you are creating a link to"
    ),
    target: str = Argument(..., help="Path to save link object"),
    source_project: Optional[str] = Option(
        None,
        "--source-project",
        "--sp",
        help="Project for source object",
        metavar=PATH_METAVAR,
    ),
    rev: Optional[str] = option_rev,
    target_project: Optional[str] = option_target_project,
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
    """Create a link (read alias) for an existing MLEM Object, including from
    remote MLEM projects.
    """
    from mlem.api.commands import link

    link(
        source=source,
        source_project=source_project,
        rev=rev,
        target=target,
        target_project=target_project,
        follow_links=follow_links,
        absolute=absolute,
    )
