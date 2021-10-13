import os
from pprint import pprint

import click
from fsspec.implementations.local import LocalFileSystem

from mlem.analytics import send_cli_call
from mlem.cli.main import mlem_command
from mlem.core.meta_io import get_fs
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    MLEM_DIR,
    MLEM_EXT,
    MlemLink,
    MlemMeta,
    find_object,
)
from mlem.utils.root import find_mlem_root


def _print_objects_of_type(path, type_):
    cls = MlemMeta.__type_map__[type_]
    fs, path = get_fs(path)
    root_path = os.path.join(
        find_mlem_root(path, fs), MLEM_DIR, cls.object_type
    )
    files = fs.glob(os.path.join(root_path, f"**{MLEM_EXT}"), recursive=True)
    if len(files) == 0:
        return
    print(type_.capitalize() + "s:")
    for file in files:
        file = file[: -len(MLEM_EXT)]
        obj_name = os.path.relpath(file, root_path)
        meta = load_meta(obj_name, follow_links=False, fs=fs)
        if (
            isinstance(meta, MlemLink)
            and obj_name != meta.mlem_link[: -len(MLEM_EXT)]
        ):
            link = f"-> {meta.mlem_link}"
        else:
            link = ""
        print("", "-", obj_name, *[link] if link else [])


TYPE_ALIASES = {
    "models": "model",
    "data": "dataset",
    "datasets": "dataset",
}


@mlem_command()
@click.argument(
    "type_",
    default="all",
)
@click.option("-r", "--repo", default=".")
def ls(type_: str, repo: str):
    """List MLEM objects of {type} in current mlem_root."""
    if type_ == "all":
        for tp in MlemMeta.subtype_mapping():
            _print_objects_of_type(repo, tp)
    else:
        type = TYPE_ALIASES.get(type_, type_)
        _print_objects_of_type(repo, type)
    send_cli_call("ls", type=type_)


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
    fs = LocalFileSystem()  # TODO: https://github.com/iterative/mlem/issues/31
    tp, _ = find_object(obj, fs)
    pprint(
        MlemMeta.subtype_mapping()[tp].read(
            obj, follow_links=follow_links, fs=fs
        )
    )
