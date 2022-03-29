from typing import List, Optional

from typer import Argument, Option

from mlem.cli.main import config_arg, mlem_command, option_repo, option_rev
from mlem.core.metadata import load_meta
from mlem.pack import Packager


@mlem_command()
def pack(
    model: str,
    out: str,
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
    """\b
    Pack model.
    Packager: either "docker_dir" or "docker".
    Out: path in case of "docker_dir" and image name in case of "docker".
    """
    from mlem.api.commands import pack

    pack(
        config_arg(Packager, load, subtype, conf, file_conf),
        load_meta(model, repo, rev),
        out,
    )
