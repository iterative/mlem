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
from mlem.core.objects import MlemModel, MlemPackager
from mlem.utils.entrypoints import list_implementations


@mlem_command("pack", section="runtime")
def pack(
    model: str = Argument(..., help="Path to model"),
    subtype: str = Argument(
        "",
        help=f"Type of packing. Choices: {list_implementations(MlemPackager)}",
        show_default=False,
    ),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    load: Optional[str] = option_load("packing"),
    conf: List[str] = option_conf("packing"),
    file_conf: List[str] = option_file_conf("packing"),
):
    """
    Pack model

    Examples:
        Build docker image from model
        $ mlem pack mymodel docker -c server.type=fastapi -c image.name=myimage

        Create pack docker_dir declaration and build it
        $ mlem create packager docker_dir -c server=fastapi -c target=build pack_dock
        $ mlem pack mymodel --load pack_dock
    """
    from mlem.api.commands import pack

    pack(
        config_arg(MlemPackager, load, subtype, conf, file_conf),
        load_meta(model, project, rev, force_type=MlemModel),
    )
