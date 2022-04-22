from typing import Optional, Type

from pydantic import BaseModel
from typer import Argument

from mlem.cli.main import mlem_command
from mlem.core.base import MlemObject, load_impl_ext
from mlem.core.objects import MlemMeta
from mlem.ext import list_implementations
from mlem.ui import EMOJI_BASE, bold, color, echo


def explain_type(cls: Type[BaseModel], prefix="", force_not_req=False):
    for name, field in sorted(
        cls.__fields__.items(), key=lambda x: not x[1].required
    ):
        if issubclass(cls, MlemMeta) and name in MlemMeta.__fields__:
            continue
        fullname = name if not prefix else f"{prefix}.{name}"
        module = field.type_.__module__
        type_name = getattr(field.type_, "__name__", str(field.type_))
        if module != "builtins" and "." not in type_name:
            type_name = f"{module}.{type_name}"
        type_name = color(type_name, "yellow")

        if field.required and not force_not_req:
            req = color("[required] ", "grey")
        else:
            req = color("[not required] ", "white")
        if not field.required:
            default = field.default
            if isinstance(default, str):
                default = f'"{default}"'
            default = f" = {default}"
        else:
            default = ""
        if (
            isinstance(field.type_, type)
            and issubclass(field.type_, MlemObject)
            and field.type_.__is_root__
        ):
            echo(
                req
                + color(fullname, "green")
                + ": One of "
                + color(f"mlem types {field.type_.abs_name}", "yellow")
            )
        elif isinstance(field.type_, type) and issubclass(
            field.type_, BaseModel
        ):
            echo(req + color(fullname, "green") + ": " + type_name)
            explain_type(field.type_, fullname, not field.required)
        else:
            echo(req + color(fullname, "green") + ": " + type_name + default)


@mlem_command("types", hidden=True)
def list_types(
    abc: Optional[str] = Argument(
        None,
        help="Subtype to list implementations. List subtypes if not provided",
    ),
    sub_type: Optional[str] = Argument(None, help="Type of `meta` subtype"),
):
    """List MLEM types implementations available in current env.
    If subtype is not provided, list ABCs

    Examples:
        List ABCs
        $ mlem types

        List available server implementations
        $ mlem types server
    """
    if abc is None:
        for at in MlemObject.abs_types.values():
            echo(EMOJI_BASE + bold(at.abs_name) + ":")
            echo(
                f"\tBase class: {at.__module__}.{at.__name__}\n\t{at.__doc__.strip()}"
            )
    elif abc == MlemMeta.abs_name:
        if sub_type is None:
            echo(list(MlemMeta.non_abstract_subtypes().keys()))
        else:
            echo(
                list_implementations(
                    MlemMeta, MlemMeta.non_abstract_subtypes()[sub_type]
                )
            )
    else:
        if sub_type is None:
            echo(list_implementations(abc))
        else:
            cls = load_impl_ext(abc, sub_type, True)
            explain_type(cls)
