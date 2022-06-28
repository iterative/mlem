from json import dumps
from pprint import pprint
from typing import List, Optional, Type

from typer import Argument, Option

from mlem.cli.main import (
    Choices,
    mlem_command,
    option_json,
    option_project,
    option_rev,
)
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


TYPE_ALIASES = {
    "models": "model",
}


@mlem_command("list", aliases=["ls"], section="common")
def ls(
    type_filter: Choices("all", *MlemObject.non_abstract_subtypes().keys()) = Option(  # type: ignore[valid-type]
        "all",
        "-t",
        "--type",
        help="Type of objects to list",
    ),
    project: str = Argument(
        "", help="Project to list from", show_default="current directory"
    ),
    rev: Optional[str] = option_rev,
    links: bool = Option(
        True, "+l/-l", "--links/--no-links", help="Include links"
    ),
    json: bool = option_json,
    ignore_errors: bool = Option(
        False, "-i", "--ignore-errors", help="Ignore corrupted objects"
    ),
):
    """List MLEM objects of in project

    Examples:
        $ mlem list https://github.com/iterative/example-mlem
        $ mlem list -t models
    """
    from mlem.api.commands import ls

    if type_filter == "all":
        types = None
    else:
        types = MlemObject.__type_map__[
            TYPE_ALIASES.get(type_filter, type_filter)
        ]

    objects = ls(
        project or ".",
        rev=rev,
        type_filter=types,
        include_links=links,
        ignore_errors=ignore_errors,
    )
    if json:
        print(
            dumps(
                {
                    cls.object_type: [obj.dict() for obj in objs]
                    for cls, objs in objects.items()
                }
            )
        )
    else:
        for cls, objs in objects.items():
            _print_objects_of_type(cls, objs)
    return {"type_filter": type_filter.value}


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
    """Print specified MLEM object

    Examples:
        Print local object
        $ mlem pprint mymodel

        Print remote object
        $ mlem pprint https://github.com/iterative/example-mlem/models/logreg
    """
    with set_echo(None if json else ...):
        meta = load_meta(
            path, project, rev, follow_links=follow_links, load_value=False
        ).dict()
    if json:
        print(dumps(meta))
    else:
        pprint(meta)
