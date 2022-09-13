from typer import Argument

from mlem.cli.main import PATH_METAVAR, mlem_command


@mlem_command("init", section="common")
def init(
    path: str = Argument(
        ".",
        help="Where to init project",
        show_default=False,
        metavar=PATH_METAVAR,
    )
):
    """Initialize a MLEM project.

    Examples:
        $ mlem init
        $ mlem init some/local/path
        $ mlem init s3://bucket/path/in/cloud
    """
    from mlem.api.commands import init

    init(path)
