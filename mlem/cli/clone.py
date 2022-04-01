from typing import Optional

from typer import Argument, Option

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_link,
    option_repo,
    option_rev,
    option_target_repo,
)


@mlem_command("clone", section="object")
def clone(
    uri: str = Argument(..., help="URI to object you want to clone"),
    target: str = Option(
        None, "-t", "--target", help="Path to store the downloaded object."
    ),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    target_repo: Optional[str] = option_target_repo,
    external: Optional[bool] = option_external,
    link: Optional[bool] = option_link,
):
    """Download MLEM object from `uri` and save it to `target`

    Examples:
        Copy remote model to local directory
        $ mlem clone models/logreg --repo https://github.com/iterative/example-mlem --rev main -t mymodel

        Copy remote model to remote MLEM repo
        $ mlem clone models/logreg --repo https://github.com/iterative/example-mlem --rev main -t mymodel --tr s3://mybucket/mymodel
    """
    from mlem.api.commands import clone

    clone(
        uri,
        target,
        repo=repo,
        rev=rev,
        target_repo=target_repo,
        external=external,
        link=link,
    )
