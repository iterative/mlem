from typer import Argument

from mlem.cli.main import PATH_METAVAR, mlem_command
from mlem.telemetry import pass_telemetry_params


@mlem_command("init", section="common")
def init(
    path: str = Argument(
        ".",
        help="Where to init project",
        show_default=False,
        metavar=PATH_METAVAR,
    )
):
    """Initialize a MLEM project."""
    from mlem.api.commands import init

    with pass_telemetry_params():
        init(path)
