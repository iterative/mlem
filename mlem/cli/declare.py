import copy
from typing import List, Optional, Type

from click import Parameter
from simple_parsing.docstring import get_attribute_docstring
from typer import Argument, Option, Typer

from ..core.base import build_mlem_object, load_impl_ext
from ..core.objects import MlemObject
from ..utils.entrypoints import list_implementations
from .main import (
    MlemCommand,
    app,
    mlem_command,
    mlem_group,
    option_external,
    option_index,
    option_project,
    wrap_build_error,
)


@mlem_command("declare", section="object")
def declare(
    object_type: str = Argument(..., help="Type of metafile to create"),
    subtype: str = Argument("", help="Subtype of MLEM object"),
    conf: Optional[List[str]] = Option(
        None,
        "-c",
        "--conf",
        help="Values for object fields in format `field.nested.name=value`",
    ),
    path: str = Argument(..., help="Where to save object"),
    project: str = option_project,
    external: bool = option_external,
    index: bool = option_index,
):
    """Creates new mlem object metafile from conf args and config files

    Examples:
        Create heroku deployment
        $ mlem declare env heroku production -c api_key=<...>
    """
    cls = MlemObject.__type_map__[object_type]
    with wrap_build_error(subtype, cls):
        meta = build_mlem_object(cls, subtype, conf, [])
    meta.dump(path, project=project, index=index, external=external)


declare2 = Typer(
    name="declare2",
    help="Manage deployments",
    cls=mlem_group("objects", aliases=["asdasd"]),
)
app.add_typer(declare2)


def get_field_help(cls: Type, field_name: str):
    return (
        get_attribute_docstring(cls, field_name).docstring_below
        or "Field docstring missing"
    )


def create_declare(typename, cls: Type[MlemObject]):
    if cls.__is_root__:
        typer = Typer(name=typename, help=cls.__doc__, cls=mlem_group("lal"))
        declare2.add_typer(typer)

        for subtype in list_implementations(MlemObject, cls):
            _how_should_i_name_this(subtype, typename, typer)
    else:

        @mlem_command(typename, parent=declare2)
        def subtype_command(
            path: str = Argument(..., help="Where to save object"),
            project: str = option_project,
            external: bool = option_external,
            index: bool = option_index,
        ):
            cls = MlemObject.__type_map__[subtype]
            with wrap_build_error(subtype, cls):
                meta = build_mlem_object(cls, subtype, [], [])
            meta.dump(path, project=project, index=index, external=external)


def _how_should_i_name_this(subtype, typename, typer):
    subtype_cls = load_impl_ext(typename, subtype)

    class SuperduperMlemCommand(MlemCommand):
        def get_help(self, ctx) -> str:
            self._getting_help = True
            return super().get_help(ctx)

        def get_params(self, ctx) -> List["Parameter"]:
            res: List[Parameter] = super().get_params(ctx)
            if hasattr(self, "_getting_help"):
                c_param = [p for p in res if p.name == "conf"][0]
                res.remove(c_param)
                for arg in subtype_cls.__fields__:
                    arg_param = copy.copy(c_param)
                    arg_param.secondary_opts = arg_param.secondary_opts + [arg]
                    arg_param.help = get_field_help(subtype_cls, arg)
                    res.append(arg_param)
            return res

    @mlem_command(subtype, parent=typer, mlem_cls=SuperduperMlemCommand)
    def subtype_command(
        path: str = Argument(..., help="Where to save object"),
        project: str = option_project,
        external: bool = option_external,
        index: bool = option_index,
        conf: Optional[List[str]] = Option(
            None,
            "-c",
            "--conf",
            help="Values for object fields in format `field.nested.name=value`",
        ),
    ):
        cls = MlemObject.__type_map__[subtype]
        with wrap_build_error(subtype, cls):
            meta = build_mlem_object(cls, subtype, conf, [])
        meta.dump(path, project=project, index=index, external=external)


for meta_type in list_implementations(MlemObject):
    create_declare(meta_type, MlemObject.__type_map__[meta_type])
