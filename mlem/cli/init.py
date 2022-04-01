from typer import Argument

from mlem.cli.main import mlem_command


@mlem_command("init", section="common")
def init(path: str = Argument(".", help="Path to repo")):
    """Initialize MLEM repo

    Examples:
        Duh
        $ mlem init

    """
    from mlem.api.commands import init

    init(path)
