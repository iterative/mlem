from typing import List, Optional

from typer import Argument, Option

from ..core.base import build_mlem_object
from ..core.objects import MlemMeta
from .main import (
    mlem_command,
    option_external,
    option_link,
    option_repo,
    wrap_build_error,
)


@mlem_command("create", section="object")
def create(
    object_type: str = Argument(..., help="Type of metafile to create"),
    subtype: str = Argument("", help="Subtype of MLEM object"),
    conf: Optional[List[str]] = Option(
        None,
        "-c",
        "--conf",
        help="Values for object fields in format `field.nested.name=value`",
    ),
    path: str = Argument(..., help="Where to save object"),
    repo: str = option_repo,
    external: bool = option_external,
    link: bool = option_link,
):
    """Creates new mlem object metafile from conf args and config files

    Examples:
        Create heroku deployment
        $ mlem create env heroku production -c api_key=<...>
    """
    cls = MlemMeta.__type_map__[object_type]
    with wrap_build_error(subtype, cls):
        meta = build_mlem_object(cls, subtype, conf, [])
    meta.dump(path, repo=repo, link=link, external=external)
