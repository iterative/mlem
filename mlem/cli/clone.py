from typing import Optional

from typer import Argument

from mlem.cli.main import (
    mlem_command,
    option_project,
    option_rev,
    option_target_project,
)


@mlem_command("clone", section="object")
def clone(
    uri: str = Argument(..., help="URI to object you want to clone"),
    target: str = Argument(..., help="Path to store the downloaded object."),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    target_project: Optional[str] = option_target_project,
):
    """Copy a MLEM Object from `uri` and
    saves a copy of it to `target` path.
    """
    from mlem.api.commands import clone

    clone(
        uri,
        target,
        project=project,
        rev=rev,
        target_project=target_project,
    )
