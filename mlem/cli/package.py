from typing import List, Optional

from typer import Argument

from mlem.cli.main import (
    config_arg,
    mlem_command,
    option_conf,
    option_file_conf,
    option_load,
    option_repo,
    option_rev,
)
from mlem.core.metadata import load_meta
from mlem.ext import list_implementations
from mlem.pack import Packager


@mlem_command("pack", section="runtime")
def pack(
    model: str = Argument(..., help="Path to model"),
    out: str = Argument(..., help="Name of the output"),
    subtype: str = Argument(
        "",
        help=f"Type of packing. Choices: {list_implementations(Packager)}",
        show_default=False,
    ),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    load: Optional[str] = option_load("packing"),
    conf: List[str] = option_conf("packing"),
    file_conf: List[str] = option_file_conf("packing"),
):
    """
    Pack model

    Examples:
        Build docker image from model
        $ mlem pack mymodel mymodel-docker-image docker -c server.type=fastapi -c image.name=myimage
    """
    from mlem.api.commands import pack

    pack(
        config_arg(Packager, load, subtype, conf, file_conf),
        load_meta(model, repo, rev),
        out,
    )
