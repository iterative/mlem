from typing import Optional

from typer import Argument, Option

from mlem.cli.main import mlem_command, option_project
from mlem.telemetry import pass_telemetry_params


@mlem_command("migrate", section="object")
def migrate(
    path: str = Argument(
        ...,
        help="URI of the MLEM object you are migrating or directory to migrate",
    ),
    project: Optional[str] = option_project,
    recursive: bool = Option(
        False, "--recursive", "-r", help="Enable recursive search of directory"
    ),
):
    """Migrate metadata objects from older MLEM version"""
    from mlem.api.migrations import migrate

    with pass_telemetry_params():
        migrate(path, project, recursive=recursive)
