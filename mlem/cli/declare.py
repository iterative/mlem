from typing import List, Optional, Type

from typer import Argument, Typer

from ..core.base import MlemABC, build_mlem_object, load_impl_ext
from ..core.objects import MlemObject
from ..utils.entrypoints import list_implementations
from .main import (
    abc_fields_parameters,
    app,
    mlem_command,
    mlem_group,
    option_conf,
    option_external,
    option_index,
    option_project,
    wrap_build_error,
)

declare = Typer(
    name="declare",
    help="""Creates new mlem object metafile from conf args and config files

    Examples:
        Create heroku deployment
        $ mlem declare env heroku production --api_key <...>
    """,
    cls=mlem_group("objects"),
    subcommand_metavar="subtype",
)
app.add_typer(declare)


def create_declare(typename, cls: Type[MlemObject]):
    if cls.__is_root__:
        typer = Typer(name=typename, help=cls.__doc__, cls=mlem_group("Other"))
        declare.add_typer(typer)

        for subtype in list_implementations(MlemObject, cls):
            create_declare_subcommand(
                typer, subtype, load_impl_ext(typename, subtype)
            )
    else:
        create_declare_subcommand(declare, typename, cls)


def create_declare_subcommand(
    parent: Typer, subtype: str, subtype_cls: Type[MlemABC]
):
    @mlem_command(
        subtype,
        parent=parent,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(subtype_cls),
    )
    def subtype_command(
        path: str = Argument(..., help="Where to save object"),
        project: str = option_project,
        external: bool = option_external,
        index: bool = option_index,
        conf: Optional[List[str]] = option_conf(),
        **__kwargs__,
    ):
        cls = subtype_cls.__type_map__[subtype]
        with wrap_build_error(subtype, cls):
            meta = build_mlem_object(cls, subtype, conf, [], **__kwargs__)
        meta.dump(path, project=project, index=index, external=external)


for meta_type in list_implementations(MlemObject):
    create_declare(meta_type, MlemObject.__type_map__[meta_type])
