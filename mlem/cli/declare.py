from typing import Type

from typer import Argument, Typer
from yaml import safe_dump

from ..core.base import MlemABC, build_mlem_object, load_impl_ext
from ..core.meta_io import Location
from ..core.objects import MlemObject
from ..utils.entrypoints import list_abstractions, list_implementations
from .main import (
    app,
    mlem_command,
    mlem_group,
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
    help="""Declares a new MLEM Object metafile from config args and config files.
    """,
    cls=mlem_group("object"),
    subcommand_metavar="subtype",
)
app.add_typer(declare)


def create_declare_mlem_object(type_name, cls: Type[MlemObject]):
    if cls.__is_root__:
        typer = Typer(
            name=type_name, help=cls.__doc__, cls=mlem_group("Mlem Objects")
        )
        declare.add_typer(typer)

        for subtype in list_implementations(MlemObject, cls):
            create_declare_mlem_object_subcommand(
                typer, subtype, type_name, cls
            )


def create_declare_mlem_object_subcommand(
    parent: Typer, subtype: str, type_name: str, parent_cls
):
    @mlem_command(
        subtype,
        section="Mlem Objects",
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
        **__kwargs__,
    ):
        subtype_cls = load_impl_ext(type_name, subtype)
        cls = subtype_cls.__type_map__[subtype]
        with wrap_build_error(subtype, cls):
            meta = build_mlem_object(
                cls, subtype, str_conf=None, file_conf=[], **__kwargs__
            )
        meta.dump(path, project=project, index=index, external=external)


for meta_type in list_implementations(MlemObject):
    create_declare_mlem_object(meta_type, MlemObject.__type_map__[meta_type])


def create_declare_mlem_abc(abs_name: str):
    try:
        root_cls = MlemABC.abs_types[abs_name]
    except KeyError:
        root_cls = None

    typer = Typer(
        name=abs_name,
        help=root_cls.__doc__
        if root_cls
        else f"Create `{abs_name}` configuration",
        cls=mlem_group("Subtypes"),
    )
    declare.add_typer(typer)

    for subtype in list_implementations(abs_name):
        if root_cls is None:
            try:
                impl = load_impl_ext(abs_name, subtype)
                root_cls = impl.__parent__  # type: ignore[assignment]
            except ImportError:
                pass
        create_declare_mlem_abc_subcommand(typer, subtype, abs_name, root_cls)


def create_declare_mlem_abc_subcommand(
    parent: Typer, subtype: str, abs_name: str, root_cls
):
    @mlem_command(
        subtype,
        section="Subtypes",
        parent=parent,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(subtype, root_cls)
        if root_cls
        else None,
        hidden=subtype.startswith("_"),
        lazy_help=lazy_class_docstring(abs_name, subtype),
    )
    def subtype_command(
        path: str = Argument(..., help="Where to save object"),
        project: str = option_project,
        **__kwargs__,
    ):
        with wrap_build_error(subtype, root_cls):
            obj = build_mlem_object(
                root_cls, subtype, str_conf=None, file_conf=[], **__kwargs__
            )
        location = Location.resolve(
            path=path, project=project, rev=None, fs=None
        )
        with location.fs.open(location.fullpath, "w") as f:
            safe_dump(obj.dict(), f)


_internal = {
    "artifact",
    "data_reader",
    "data_type",
    "data_writer",
    "deploy_state",
    "import",
    "interface",
    "meta",
    "model_io",
    "model_type",
    "requirement",
    "resolver",
    "storage",
    "state",
}
for abs_name in list_abstractions(include_hidden=False):
    if abs_name in {"builder", "env", "deployment"}:
        continue
    if abs_name in _internal:
        continue
    create_declare_mlem_abc(abs_name)
