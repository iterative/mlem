from typing import Iterator, Optional, Type

from pydantic import BaseModel
from typer import Argument

from mlem.cli.main import mlem_command
from mlem.cli.utils import CliTypeField, iterate_type_fields, parse_type_field
from mlem.core.base import MlemABC, load_impl_ext
from mlem.core.errors import MlemError
from mlem.core.objects import MlemObject
from mlem.ui import EMOJI_BASE, bold, color, echo
from mlem.utils.entrypoints import list_abstractions, list_implementations


def _add_examples(
    generator: Iterator[CliTypeField],
    root_cls: Type[BaseModel],
    parent_help=None,
):
    for field in generator:
        field.help = parent_help or field.help
        yield field
        if field.is_list or field.is_mapping:
            key = ".key" if field.is_mapping else ".0"
            yield from _add_examples(
                parse_type_field(
                    path=field.path + key,
                    type_=field.type_,
                    help_=field.help,
                    is_list=False,
                    is_mapping=False,
                    required=False,
                    allow_none=False,
                    default=None,
                    root_cls=root_cls,
                ),
                root_cls=root_cls,
                parent_help=f"Element of {field.path}",
            )


def type_fields_with_collection_examples(cls):
    yield from _add_examples(iterate_type_fields(cls), root_cls=cls)


def explain_type(cls: Type[BaseModel]):
    echo(
        color("Type ", "")
        + color(cls.__module__ + ".", "yellow")
        + color(cls.__name__, "green")
    )
    if issubclass(cls, MlemABC):
        echo(color("MlemABC parent type: ", "") + color(cls.abs_name, "green"))
        echo(color("MlemABC type: ", "") + color(cls.__get_alias__(), "green"))
    if issubclass(cls, MlemObject):
        echo(
            color("MlemObject type name: ", "")
            + color(cls.object_type, "green")
        )
    echo((cls.__doc__ or "Class docstring missing").strip())
    fields = list(type_fields_with_collection_examples(cls))
    if not fields:
        echo("No fields")
    else:
        echo("Fields:")
    for field in fields:
        echo(field.to_text())


@mlem_command("types", hidden=True)
def list_types(
    abc: Optional[str] = Argument(
        None,
        help="Subtype to list implementations. List subtypes if not provided.",
    ),
    sub_type: Optional[str] = Argument(None, help="Type of `meta` subtype"),
):
    """List different implementations available for a particular MLEM type. If a
    subtype is not provided, list all available MLEM types.
    """
    if abc is None:
        for at in MlemABC.abs_types.values():
            echo(EMOJI_BASE + bold(at.abs_name) + ":")
            echo(
                f"\tBase class: {at.__module__}.{at.__name__}\n\t{(at.__doc__ or 'Class docstring missing').strip()}"
            )
    elif abc == MlemObject.abs_name:
        if sub_type is None:
            echo("\n".join(MlemObject.non_abstract_subtypes().keys()))
        else:
            mlem_object_type = MlemObject.non_abstract_subtypes()[sub_type]
            if mlem_object_type.__is_root__:
                echo(
                    "\n".join(
                        list_implementations(
                            MlemObject, mlem_object_type, include_hidden=False
                        )
                    )
                )
            else:
                explain_type(mlem_object_type)
    else:
        if sub_type is None:
            abcs = list_abstractions(include_hidden=False)
            if abc not in abcs:
                raise MlemError(
                    f"Unknown abc \"{abc}\". Known abcs: {' '.join(abcs)}"
                )
            echo("\n".join(list_implementations(abc, include_hidden=False)))
        else:
            try:
                cls = load_impl_ext(abc, sub_type, True)
            except ValueError as e:
                raise MlemError(
                    f"Unknown implementation \"{sub_type}\" of abc \"{abc}\". Known implementations: {' '.join(list_implementations(abc, include_hidden=False))}"
                ) from e
            explain_type(cls)
