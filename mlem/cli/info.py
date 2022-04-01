from pprint import pprint
from typing import List, Optional, Type

from typer import Argument, Option

from mlem.cli.main import mlem_command, option_repo, option_rev
from mlem.core.metadata import load_meta
from mlem.core.objects import MLEM_EXT, MlemLink, MlemMeta
from mlem.ui import echo


def _print_objects_of_type(cls: Type[MlemMeta], objects: List[MlemMeta]):
    if len(objects) == 0:
        return

    echo(cls.object_type.capitalize() + "s:")
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
    "data": "dataset",
    "datasets": "dataset",
}


@mlem_command("list", section="common")
def ls(
    type_filter: str = Argument("all", help="Type of objects to list"),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    links: bool = Option(
        True, "+l/-l", "--links/--no-links", help="Include links"
    ),
):
    """List MLEM objects of in repo

    Examples:
        $ mlem list --repo https://github.com/iterative/example-mlem
    """
    from mlem.api.commands import ls

    if type_filter == "all":
        types = None
    else:
        types = MlemMeta.__type_map__[
            TYPE_ALIASES.get(type_filter, type_filter)
        ]

    objects = ls(repo or ".", rev=rev, type_filter=types, include_links=links)
    for cls, objs in objects.items():
        _print_objects_of_type(cls, objs)
    return {"type_filter": type_filter}


@mlem_command("pprint", hidden=True)
def pretty_print(
    path: str = Argument(..., help="Path to object"),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    follow_links: bool = Option(
        False,
        "-f",
        "--follow-links",
        help="If specified, follow the link to the actual object.",
    ),
):
    """Print specified MLEM object

    Examples:
        Print local object
        $ mlem pprint mymodel

        Print remote object
        $ mlem pprint https://github.com/iterative/example-mlem/models/logreg
    """
    pprint(
        load_meta(
            path, repo, rev, follow_links=follow_links, load_value=False
        ).dict()
    )
