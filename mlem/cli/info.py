import os
from pprint import pprint
from typing import List, Type

import click

from mlem.cli.main import mlem_command
from mlem.core.meta_io import get_fs
from mlem.core.objects import (
    MLEM_DIR,
    MLEM_EXT,
    MlemLink,
    MlemMeta,
    find_object,
)
from mlem.utils.root import find_mlem_root


def _print_objects_of_type(
    cls: Type[MlemMeta], objects: List[MlemMeta], mlem_root: str
):
    if len(objects) == 0:
        return

    print(cls.object_type.capitalize() + "s:")
    for meta in objects:
        obj_name = os.path.relpath(meta.name or "?", mlem_root)
        if (
            isinstance(meta, MlemLink)
            and obj_name != meta.mlem_link[: -len(MLEM_EXT)]
        ):
            link = f"-> {os.path.dirname(meta.mlem_link)}"
            obj_name = os.path.relpath(
                obj_name, os.path.join(MLEM_DIR, cls.object_type)
            )[: -len(MLEM_EXT)]
        else:
            link = ""
            obj_name = os.path.dirname(obj_name)
        print("", "-", obj_name, *[link] if link else [])


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
@click.option("-r", "--repo", default=".")
@click.option("+l/-l", "--links/--no-links", default=True, is_flag=True)
def ls(type_filter: str, repo: str, links: bool):
    """List MLEM objects of {type} in current mlem_root."""
    from mlem.api.commands import ls

    if type_filter == "all":
        types = None
    else:
        types = MlemMeta.__type_map__[
            TYPE_ALIASES.get(type_filter, type_filter)
        ]

    objects = ls(repo, types, include_links=links)
    fs, path = get_fs(repo)
    mlem_root = find_mlem_root(path, fs)
    for cls, objs in objects.items():
        _print_objects_of_type(cls, objs, mlem_root)
    return {"type_filter": type_filter}


@mlem_command("pprint")
@click.argument("obj")
@click.option(
    "-f",
    "--follow-links",
    default=False,
    type=click.BOOL,
    is_flag=True,
    help="If specified, follow the link to the actual object.",
)
def pretty_print(obj: str, follow_links: bool):
    """Print __str__ for the specified MLEM object."""
    fs, path = get_fs(obj)
    tp, _ = find_object(path, fs)
    pprint(
        MlemMeta.subtype_mapping()[tp].read(
            path, follow_links=follow_links, fs=fs
        )
    )
