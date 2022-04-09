from typer import Argument

from mlem.cli.main import mlem_command


@mlem_command("init", section="common")
def init(
    path: str = Argument(".", help="Where to init repo", show_default=False)
):
    """Initialize MLEM repo

    Examples:
        $ mlem init
        $ mlem init some/local/path
        $ mlem init s3://bucket/path/in/cloud
    """
    from mlem.api.commands import init

    init(path)
