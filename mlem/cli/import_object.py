from typing import Optional

from typer import Argument, Option

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_index,
    option_project,
    option_rev,
    option_target_project,
)
from mlem.core.import_objects import ImportHook
from mlem.utils.entrypoints import list_implementations


@mlem_command("import", section="object")
def import_object(
    uri: str = Argument(..., help="File to import"),
    target: str = Argument(..., help="Path to save MLEM object"),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    target_project: Optional[str] = option_target_project,
    copy: bool = Option(
        True,
        help="Whether to create a copy of file in target location or just link existing file",
    ),
    type_: Optional[str] = Option(None, "--type", help=f"Specify how to read file Available types: {list_implementations(ImportHook)}", show_default="auto infer"),  # type: ignore
    index: bool = option_index,
    external: bool = option_external,
):
    """Create MLEM model or data metadata from file/dir

    Examples:
        Create MLEM data from local csv
        $ mlem import data/data.csv data/imported_data --type pandas[csv]

        Create MLEM model from local pickle file
        $ mlem import data/model.pkl data/imported_model

        Create MLEM model from remote pickle file
        $ mlem import models/logreg --project https://github.com/iterative/example-mlem --rev no-dvc data/imported_model --type pickle
    """
    from mlem.api.commands import import_object

    import_object(
        uri,
        project=project,
        rev=rev,
        target=target,
        target_project=target_project,
        copy_data=copy,
        type_=type_,
        external=external,
        index=index,
    )
