from typing import List, Optional

from typer import Argument, Option, echo

from mlem.cli.main import config_arg, mlem_command, option_repo, option_rev
from mlem.core.metadata import load_meta
from mlem.core.objects import ModelMeta
from mlem.runtime.server.base import Server


@mlem_command("serve")
def serve(
    model: str,
    subtype: str = Argument(""),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    load: Optional[str] = Option(
        None,
        "-l",
        "--load",
    ),
    conf: List[str] = Option(None, "-c", "--conf"),
    file_conf: List[str] = Option(None, "-f", "--file_conf"),
):
    """Serve selected model."""
    from mlem.api.commands import serve

    echo("Serving")
    serve(
        load_meta(model, repo, rev, force_type=ModelMeta),
        config_arg(Server, load, subtype, conf, file_conf),
    )
