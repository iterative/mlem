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


@mlem_command("import", section="object")
def import_object(
    uri: str = Argument(..., help="File to import"),
    target: str = Argument(..., help="Path whare to save MLEM object"),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    target_repo: Optional[str] = option_target_repo,
    copy: bool = Option(
        True,
        help="Whether to create a copy of file in target location or just link existing file",
    ),
    type_: Optional[str] = Option(None, "--type", help="Specify how to read file", show_default="auto infer"),  # type: ignore
    link: bool = option_link,
    external: bool = option_external,
):
    """Create MLEM model or dataset metadata from file/dir

    Examples:
        Create MLEM dataset from local csv
        $ mlem import data/data.csv data/imported_data --type pandas[csv]

        Create MLEM model from local pickle file
        $ mlem import data/model.pkl data/imported_model

        Create MLEM model from remote pickle file
        $ mlem import models/logreg --repo https://github.com/iterative/example-mlem --rev no-dvc data/imported_model --type pickle
    """
    from mlem.api.commands import import_object

    import_object(
        uri,
        repo=repo,
        rev=rev,
        target=target,
        target_repo=target_repo,
        copy_data=copy,
        type_=type_,
        external=external,
        link=link,
    )
