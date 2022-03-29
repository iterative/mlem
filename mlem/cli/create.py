from typing import List, Optional

from typer import Argument, Option

from ..core.base import build_mlem_object
from ..core.objects import MlemMeta
from .main import mlem_command, option_external, option_link, option_repo


@mlem_command("create")
def create(
    object_type: str = Argument(
        ...,
    ),
    subtype: str = Argument(""),
    conf: Optional[List[str]] = Option(
        None,
        "-c",
        "--conf",
    ),
    path: str = Argument(...),
    repo: str = option_repo,
    external: bool = option_external,
    link: bool = option_link,
):
    """Creates new mlem object metafile from conf args and config files

    Example: mlem create env heroku -c api_key=<...>"""
    cls = MlemMeta.__type_map__[object_type]
    meta = build_mlem_object(cls, subtype, conf, [])
    meta.dump(path, repo=repo, link=link, external=external)
