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
from mlem.core.objects import ModelMeta
from mlem.ext import list_implementations
from mlem.runtime.server.base import Server


@mlem_command("serve", section="runtime")
def serve(
    model: str = Argument(..., help="Model to create service from"),
    subtype: str = Argument(
        "", help=f"Server type. Choices: {list_implementations(Server)}"
    ),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    load: Optional[str] = option_load("server"),
    conf: List[str] = option_conf("server"),
    file_conf: List[str] = option_file_conf("server"),
):
    """Serve selected model

    Examples:
        $ mlem serve https://github.com/iterative/example-mlem/models/logreg fastapi
    """
    from mlem.api.commands import serve

    serve(
        load_meta(model, repo, rev, force_type=ModelMeta),
        config_arg(Server, load, subtype, conf, file_conf),
    )
