from typing import Optional

from typer import Argument

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_index,
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
    external: Optional[bool] = option_external,
    index: Optional[bool] = option_index,
):
    """Download MLEM object from `uri` and save it to `target`

    Examples:
        Copy remote model to local directory
        $ mlem clone models/logreg --project https://github.com/iterative/example-mlem --rev main mymodel

        Copy remote model to remote MLEM project
        $ mlem clone models/logreg --project https://github.com/iterative/example-mlem --rev main mymodel --tp s3://mybucket/mymodel
    """
    from mlem.api.commands import clone

    clone(
        uri,
        target,
        project=project,
        rev=rev,
        target_project=target_project,
        external=external,
        index=index,
    )
