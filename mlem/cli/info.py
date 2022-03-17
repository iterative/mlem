from pprint import pprint
from typing import List, Optional, Type

import click

from mlem.cli.main import mlem_command, option_repo, option_rev
from mlem.core.metadata import load_meta
from mlem.core.objects import MLEM_EXT, MlemLink, MlemMeta


def _print_objects_of_type(cls: Type[MlemMeta], objects: List[MlemMeta]):
    if len(objects) == 0:
        return

    print(cls.object_type.capitalize() + "s:")
    for meta in objects:
        if (
            isinstance(meta, MlemLink)
            and meta.name != meta.path[: -len(MLEM_EXT)]
        ):
            link = f"-> {meta.path[:-len(MLEM_EXT)]}"
        else:
            link = ""
        print("", "-", meta.name, *[link] if link else [])


TYPE_ALIASES = {
    "models": "model",
    "data": "dataset",
    "datasets": "dataset",
}


@mlem_command()
@click.argument(
    "type_filter",
    default="all",
)
@option_repo
@option_rev
@click.option("+l/-l", "--links/--no-links", default=True, is_flag=True)
def ls(type_filter: str, repo: str, rev: Optional[str], links: bool):
    """List MLEM objects of {type} in repo."""
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


@mlem_command("pprint")
@click.argument("path")
@click.option(
    "-f",
    "--follow-links",
    default=False,
    type=click.BOOL,
    is_flag=True,
    help="If specified, follow the link to the actual object.",
)
@option_repo
@option_rev
def pretty_print(
    path: str, repo: str = None, rev: str = None, follow_links: bool = False
):
    """Print __str__ for the specified MLEM object."""
    pprint(
        load_meta(path, repo, rev, follow_links=follow_links, load_value=False)
    )
