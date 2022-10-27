from typing import Any, Dict, Type

from typer import Argument, Typer
from yaml import safe_dump

from ..core.base import MlemABC, build_mlem_object, load_impl_ext
from ..core.meta_io import Location
from ..core.objects import EnvLink, MlemDeployment, MlemObject
from ..utils.entrypoints import list_abstractions, list_implementations
from .main import app, mlem_command, mlem_group, option_project
from .utils import (
    NOT_SET,
    CallContext,
    CliTypeField,
    _option_from_field,
    _options_from_model,
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


def add_env_params_deployment(subtype, parent_cls: Type[MlemDeployment]):
    try:
        impl = load_impl_ext(parent_cls.object_type, subtype)
    except ImportError:
        return lambda ctx: []

    assert issubclass(impl, MlemDeployment)  # just to help mypy
    env_impl = impl.env_type

    def add_env(ctx: CallContext):
        yield from abc_fields_parameters(subtype, parent_cls)(ctx)
        yield from (
            _options_from_model(env_impl, ctx, path="env", force_not_set=True)
        )
        yield from (
            _options_from_model(EnvLink, ctx, path="env", force_not_set=True)
        )
        yield _option_from_field(
            CliTypeField(
                path="env",
                required=False,
                allow_none=False,
                type_=str,
                help="",
                default=NOT_SET,
                is_list=False,
                is_mapping=False,
            ),
            "env",
        )

    return add_env


def process_env_params_deployments(
    subtype, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    env_params = {p[len("env.") :] for p in kwargs if p.startswith("env.")}
    if not env_params.issubset({"project", "path", "rev"}):
        kwargs["env"] = subtype
    return kwargs


_add_fields = {"deployment": add_env_params_deployment}
_process_fields = {"deployment": process_env_params_deployments}


def add_fields(subtype: str, parent_cls):
    return _add_fields.get(parent_cls.object_type, abc_fields_parameters)(
        subtype, parent_cls
    )


def process_fields(subtype: str, parent_cls, kwargs):
    if parent_cls.object_type in _process_fields:
        kwargs = _process_fields[parent_cls.object_type](subtype, kwargs)
    return kwargs


def create_declare_mlem_object_subcommand(
    parent: Typer, subtype: str, type_name: str, parent_cls
):
    @mlem_command(
        subtype,
        section="MLEM Objects",
        parent=parent,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=add_fields(subtype, parent_cls),
        hidden=subtype.startswith("_"),
        lazy_help=lazy_class_docstring(type_name, subtype),
    )
    def subtype_command(
        path: str = Argument(
            ..., help="Where to save the object (.mlem file)"
        ),
        project: str = option_project,
        **__kwargs__,
    ):
        __kwargs__ = process_fields(subtype, parent_cls, __kwargs__)
        subtype_cls = load_impl_ext(type_name, subtype)
        cls = subtype_cls.__type_map__[subtype]
        with wrap_build_error(subtype, cls):
            meta = build_mlem_object(
                cls, subtype, str_conf=None, file_conf=[], **__kwargs__
            )
        meta.dump(path, project=project)


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


_exposed = {"server", "client", "docker_registry"}
for abs_name in list_abstractions(include_hidden=False):
    if abs_name not in _exposed:
        continue
    create_declare_mlem_abc(abs_name)
