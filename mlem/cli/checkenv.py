from typing import Optional

from typer import Argument

from mlem.cli.main import (
    PATH_METAVAR,
    mlem_command,
    option_project,
    option_rev,
)
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemData, MlemModel
from mlem.ui import EMOJI_OK, echo


@mlem_command("checkenv", hidden=True)
def checkenv(
    path: str = Argument(..., help="Path to object", metavar=PATH_METAVAR),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
):
    """Check that current environment satisfies object requrements"""
    meta = load_meta(path, project, rev, follow_links=True, load_value=False)
    if isinstance(meta, (MlemModel, MlemData)):
        meta.checkenv()
    echo(EMOJI_OK + "Requirements are satisfied!")
