from typing import Optional

from typer import Argument, Option

from mlem.cli.main import (
    mlem_command,
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
):
    """Create a `.mlem` metafile for a model or data in any file or directory."""
    from mlem.api.commands import import_object

    import_object(
        uri,
        project=project,
        rev=rev,
        target=target,
        target_project=target_project,
        copy_data=copy,
        type_=type_,
    )
