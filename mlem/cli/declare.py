from typing import List, Optional, Type

from typer import Argument, Typer

from ..core.base import build_mlem_object, load_impl_ext
from ..core.objects import MlemObject
from ..utils.entrypoints import list_implementations
from .main import (
    app,
    mlem_command,
    mlem_group,
    option_conf,
    option_external,
    option_index,
    option_project,
)
from .utils import (
    abc_fields_parameters,
    lazy_class_docstring,
    wrap_build_error,
)

declare = Typer(
    name="declare",
    help="""Creates new mlem object metafile from conf args and config files

    Examples:
        Create heroku deployment
        $ mlem declare env heroku production --api_key <...>
    """,
    cls=mlem_group("object"),
    subcommand_metavar="subtype",
)
app.add_typer(declare)


def create_declare(type_name, cls: Type[MlemObject]):
    if cls.__is_root__:
        typer = Typer(
            name=type_name, help=cls.__doc__, cls=mlem_group("Subtypes")
        )
        declare.add_typer(typer)

        for subtype in list_implementations(MlemObject, cls):
            create_declare_subcommand(typer, subtype, type_name, cls)


def create_declare_subcommand(
    parent: Typer, subtype: str, type_name: str, parent_cls
):
    @mlem_command(
        subtype,
        section="Subtypes",
        parent=parent,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(subtype, parent_cls),
        hidden=subtype.startswith("_"),
        lazy_help=lazy_class_docstring(type_name, subtype),
    )
    def subtype_command(
        path: str = Argument(..., help="Where to save object"),
        project: str = option_project,
        external: bool = option_external,
        index: bool = option_index,
        conf: Optional[List[str]] = option_conf(),
        **__kwargs__,
    ):
        subtype_cls = load_impl_ext(type_name, subtype)
        cls = subtype_cls.__type_map__[subtype]
        with wrap_build_error(subtype, cls):
            meta = build_mlem_object(cls, subtype, conf, [], **__kwargs__)
        meta.dump(path, project=project, index=index, external=external)


for meta_type in list_implementations(MlemObject):
    create_declare(meta_type, MlemObject.__type_map__[meta_type])
