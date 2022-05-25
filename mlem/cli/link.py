from typing import Optional

from typer import Argument, Option

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_rev,
    option_target_project,
)


@mlem_command("link", section="object")
def link(
    source: str = Argument(..., help="URI to object you are crating link to"),
    target: str = Argument(..., help="Path to save link object"),
    source_project: Optional[str] = Option(
        None,
        "--source-project",
        "--sp",
        help="Project for source object",
    ),
    rev: Optional[str] = option_rev,
    target_project: Optional[str] = option_target_project,
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
    """Create link for MLEM object

    Examples:
        Add alias to local object
        $ mlem link my_model latest

        Add remote object to your project without copy
        $ mlem link models/logreg --source-project https://github.com/iteartive/example-mlem remote_model
    """
    from mlem.api.commands import link

    link(
        source=source,
        source_project=source_project,
        rev=rev,
        target=target,
        target_project=target_project,
        follow_links=follow_links,
        external=external or False,
        absolute=absolute,
    )
