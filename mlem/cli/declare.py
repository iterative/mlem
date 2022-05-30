from typing import List, Optional

from typer import Argument, Option

from ..core.base import build_mlem_object
from ..core.objects import MlemObject
from .main import (
    mlem_command,
    option_external,
    option_index,
    option_project,
    wrap_build_error,
)


@mlem_command("declare", section="object")
def declare(
    object_type: str = Argument(..., help="Type of metafile to create"),
    subtype: str = Argument("", help="Subtype of MLEM object"),
    conf: Optional[List[str]] = Option(
        None,
        "-c",
        "--conf",
        help="Values for object fields in format `field.nested.name=value`",
    ),
    path: str = Argument(..., help="Where to save object"),
    project: str = option_project,
    external: bool = option_external,
    index: bool = option_index,
):
    """Creates new mlem object metafile from conf args and config files

    Examples:
        Create heroku deployment
        $ mlem declare env heroku production -c api_key=<...>
    """
    cls = MlemObject.__type_map__[object_type]
    with wrap_build_error(subtype, cls):
        meta = build_mlem_object(cls, subtype, conf, [])
    meta.dump(path, project=project, index=index, external=external)
