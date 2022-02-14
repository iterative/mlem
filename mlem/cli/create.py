from typing import List

import click

from ..core.base import build_mlem_object
from ..core.objects import MlemMeta
from .main import mlem_command, option_external, option_link, option_repo


@mlem_command("create")
@click.argument("object_type")
@click.argument("subtype", default="")
@click.option("-c", "--conf", multiple=True)
@click.argument("path")
@option_repo
@option_external
@option_link
def create(
    object_type: str,
    subtype: str,
    conf: List[str],
    path: str,
    repo: str,
    external: bool,
    link: bool,
):
    """Creates new mlem object metafile from conf args and config files

    Example: mlem create env heroku -c api_key=<...>"""
    cls = MlemMeta.__type_map__[object_type]
    meta = build_mlem_object(cls, subtype, conf, [])
    meta.dump(path, repo=repo, link=link, external=external)
