from typer import Argument

from mlem.cli.main import mlem_command


@mlem_command("init", section="common")
def init(
    path: str = Argument(".", help="Where to init project", show_default=False)
):
    """Initialize a MLEM project."""
    from mlem.api.commands import init

    init(path)
