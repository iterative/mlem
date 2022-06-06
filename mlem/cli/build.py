from typing import List, Optional

from typer import Argument

from mlem.cli.main import (
    config_arg,
    mlem_command,
    option_conf,
    option_file_conf,
    option_load,
    option_project,
    option_rev,
)
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.utils.entrypoints import list_implementations


@mlem_command("build", section="runtime", aliases=["export"])
def build(
    model: str = Argument(..., help="Path to model"),
    subtype: str = Argument(
        "",
        help=f"Type of build. Choices: {list_implementations(MlemBuilder)}",
        show_default=False,
    ),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    load: Optional[str] = option_load("builder"),
    conf: List[str] = option_conf("builder"),
    file_conf: List[str] = option_file_conf("builder"),
):
    """
    Build/export model

    Examples:
        Build docker image from model
        $ mlem build mymodel docker -c server.type=fastapi -c image.name=myimage

        Create build docker_dir declaration and build it
        $ mlem declare builder docker_dir -c server=fastapi -c target=build build_dock
        $ mlem build mymodel --load build_dock
    """
    from mlem.api.commands import build

    build(
        config_arg(MlemBuilder, load, subtype, conf, file_conf),
        load_meta(model, project, rev, force_type=MlemModel),
    )
