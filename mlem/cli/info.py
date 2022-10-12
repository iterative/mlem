from json import dumps
from pprint import pprint
from typing import List, Optional, Type

from typer import Argument, Option

from mlem.cli.main import mlem_command, option_json, option_project, option_rev
from mlem.core.metadata import load_meta
from mlem.core.objects import MLEM_EXT, MlemLink, MlemObject
from mlem.ui import echo, set_echo

OBJECT_TYPE_NAMES = {"data": "Data"}


def _print_objects_of_type(cls: Type[MlemObject], objects: List[MlemObject]):
    if len(objects) == 0:
        return

    echo(
        OBJECT_TYPE_NAMES.get(
            cls.object_type, cls.object_type.capitalize() + "s"
        )
        + ":"
    )
    for meta in objects:
        if (
            isinstance(meta, MlemLink)
            and meta.name != meta.path[: -len(MLEM_EXT)]
        ):
            link = f"-> {meta.path[:-len(MLEM_EXT)]}"
        else:
            link = ""
        echo("", "-", meta.name, *[link] if link else [])


@mlem_command("pprint", hidden=True)
def pretty_print(
    path: str = Argument(..., help="Path to object"),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    follow_links: bool = Option(
        False,
        "-f",
        "--follow-links",
        help="If specified, follow the link to the actual object.",
    ),
    json: bool = option_json,
):
    """Display all details about a specific MLEM Object from an existing MLEM
    project.
    """
    with set_echo(None if json else ...):
        meta = load_meta(
            path, project, rev, follow_links=follow_links, load_value=False
        ).dict()
    if json:
        print(dumps(meta))
    else:
        pprint(meta)
