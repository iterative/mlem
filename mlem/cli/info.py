import glob
import os
from pprint import pprint

import click
from fsspec.implementations.local import LocalFileSystem

from mlem.cli.main import cli
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    MLEM_DIR,
    MLEM_EXT,
    MlemLink,
    MlemMeta,
    find_object,
)
from mlem.utils.root import find_mlem_root


def _print_objects_of_type(type):
    cls = MlemMeta.__type_map__[type]
    root_path = os.path.join(find_mlem_root(), MLEM_DIR, cls.object_type)
    files = glob.glob(
        os.path.join(root_path, "**", f"*{MLEM_EXT}"), recursive=True
    )
    if len(files) == 0:
        return
    print(type.capitalize() + "s:")
    for file in files:
        file = file[: -len(MLEM_EXT)]
        obj_name = os.path.relpath(file, root_path)
        meta = load_meta(obj_name, follow_links=False)
        if (
            isinstance(meta, MlemLink)
            and obj_name != meta.mlem_link[: -len(MLEM_EXT)]
        ):
            link = f"-> {meta.mlem_link}"
        else:
            link = ""
        print("", "-", obj_name, link)


TYPE_ALIASES = {
    "models": "model",
    "data": "dataset",
    "datasets": "dataset",
}


@cli.command()
@click.argument("type", default="all")
def ls(type: str):
    """List MLEM objects in current mlem_root."""
    if type == "all":
        for tp in MlemMeta.subtype_mapping().keys():
            _print_objects_of_type(tp)
    else:
        type = TYPE_ALIASES.get(type, type)
        _print_objects_of_type(type)


@cli.command("pprint")
@click.argument("obj")
@click.option(
    "-f", "--follow-links", default=False, type=click.BOOL, is_flag=True
)
def pretty_print(obj: str, follow_links: bool):
    """Print __str__ for the specified MLEM object."""
    fs = LocalFileSystem()  # TODO: https://github.com/iterative/mlem/issues/31
    tp, path = find_object(obj, fs)
    pprint(
        MlemMeta.subtype_mapping()[tp].read(
            obj, follow_links=follow_links, fs=fs
        )
    )
