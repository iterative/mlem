from typer import Argument

from mlem.cli.main import mlem_command


@mlem_command("init")
def init(path: str = Argument(".")):
    """Create .mlem folder in {path}"""
    from mlem.api.commands import init

    init(path)
