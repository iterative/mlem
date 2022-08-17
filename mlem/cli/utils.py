import ast
import contextlib
import inspect
from enum import Enum, EnumMeta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import typer
from pydantic import BaseModel, MissingError, ValidationError, parse_obj_as
from pydantic.error_wrappers import ErrorWrapper
from pydantic.fields import (
    MAPPING_LIKE_SHAPES,
    SHAPE_LIST,
    SHAPE_SEQUENCE,
    SHAPE_SET,
    SHAPE_TUPLE,
    SHAPE_TUPLE_ELLIPSIS,
)
from pydantic.typing import get_args
from typer.core import TyperOption
from yaml import safe_load

from mlem import LOCAL_CONFIG
from mlem.core.base import MlemABC, build_mlem_object, load_impl_ext
from mlem.core.errors import ExtensionRequirementError
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemObject
from mlem.ui import EMOJI_FAIL, color
from mlem.utils.entrypoints import list_implementations
from mlem.utils.module import lstrip_lines

LIST_LIKE_SHAPES = (
    SHAPE_LIST,
    SHAPE_TUPLE,
    SHAPE_SET,
    SHAPE_TUPLE_ELLIPSIS,
    SHAPE_SEQUENCE,
)


class ChoicesMeta(EnumMeta):
    def __call__(cls, *names, module=None, qualname=None, type=None, start=1):
        if len(names) == 1:
            return super().__call__(names[0])
        return super().__call__(
            "Choice",
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )


class Choices(str, Enum, metaclass=ChoicesMeta):
    def _generate_next_value_(  # pylint: disable=no-self-argument
        name, start, count, last_values
    ):
        return name


class CliTypeField(BaseModel):
    required: bool
    path: str
    type_: Type
    help: str
    default: Any
    is_list: bool
    is_mapping: bool
    mapping_key_type: Optional[Type]

    @property
    def type_repr(self):
        type_name = self.type_.__name__
        if self.is_list:
            return f"List[{type_name}]"
        if self.is_mapping:
            return f"Dict[{self.mapping_key_type.__name__}, {type_name}]"
        return type_name

    def to_text(self):
        req = (
            color("[required]", "")
            if self.required
            else color("[not required]", "white")
        )
        if not self.required:
            default = self.default
            if isinstance(default, str):
                default = f'"{default}"'
            default = f" = {default}"
        else:
            default = ""
        return (
            req
            + " "
            + color(self.path, "green")
            + ": "
            + self.type_repr
            + default
            + "\n\t"
            + self.help
        )


@lru_cache()
def get_attribute_docstrings(cls) -> Dict[str, str]:
    res = {}
    tree = ast.parse(lstrip_lines(inspect.getsource(cls)))
    class_def = tree.body[0]
    assert isinstance(class_def, ast.ClassDef)
    field: Optional[str] = None
    for statement in class_def.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            field = statement.target.id
            continue
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            field = statement.targets[0].id
            continue
        if (
            field is not None
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            res[field] = statement.value.value
        field = None
    return res


@lru_cache()
def get_field_help(cls: Type, field_name: str):
    for base_cls in cls.mro():
        if base_cls is object:
            continue
        docsting = get_attribute_docstrings(base_cls).get(field_name)
        if docsting:
            return docsting
    return "Field docstring missing"


def iterate_type_fields(cls: Type[BaseModel], prefix="", force_not_req=False):
    for name, field in sorted(
        cls.__fields__.items(), key=lambda x: not x[1].required
    ):
        name = field.alias or name
        if issubclass(cls, MlemObject) and name in MlemObject.__fields__:
            continue
        if (
            issubclass(cls, MlemABC)
            and name in cls.__config__.exclude
            or field.field_info.exclude
        ):
            continue
        if name == "__root__":
            fullname = prefix
        else:
            fullname = name if not prefix else f"{prefix}.{name}"

        req = field.required and not force_not_req
        default = field.default
        docstring = get_field_help(cls, name)
        field_type = field.type_
        if not isinstance(field_type, type):
            # typing.GenericAlias
            generic_args = get_args(field_type)
            if len(generic_args) > 0:
                field_type = field_type.__args__[0]
            else:
                field_type = object
        if (
            isinstance(field_type, type)
            and issubclass(field_type, MlemABC)
            and field_type.__is_root__
        ):
            if isinstance(default, field_type):
                default = default.__get_alias__()
            yield CliTypeField(
                required=req,
                path=fullname,
                type_=field_type,
                help=f"{docstring}. One of {list_implementations(field_type, include_hidden=False)}. Run 'mlem types {field_type.abs_name} <subtype>' for list of nested fields for each subtype",
                default=default,
                is_list=field.shape in LIST_LIKE_SHAPES,
                is_mapping=field.shape in MAPPING_LIKE_SHAPES,
                mapping_key_type=str,
            )
        elif isinstance(field_type, type) and issubclass(
            field_type, BaseModel
        ):
            yield from iterate_type_fields(
                field_type, fullname, not field.required
            )
        else:
            yield CliTypeField(
                required=req,
                path=fullname,
                type_=field_type,
                default=default,
                help=docstring,
                is_list=field.shape in LIST_LIKE_SHAPES,
                is_mapping=field.shape in MAPPING_LIKE_SHAPES,
                mapping_key_type=str,
            )


def _options_from_cls(cls: Type[MlemABC], params: Dict, prefix=""):
    for field in iterate_type_fields(cls, prefix=prefix):
        type_ = field.type_
        if issubclass(type_, MlemABC) and type_.__is_root__:
            if field.path in params:
                yield from _options_from_cls(
                    load_impl_ext(type_.abs_name, params[field.path]),
                    params,
                    field.path,
                )
            type_ = str
        if type_ is object:
            # TODO: dicts and lists
            continue
        option = TyperOption(
            param_decls=[f"--{field.path}", field.path.replace(".", "_")],
            type=type_,
            required=field.required,
            default=field.default,
            help=field.help,
            show_default=not field.required,
            multiple=field.is_list,
        )
        option.name = field.path
        yield option


def abc_fields_parameters(type_name: str, mlem_abc: Type[MlemABC]):
    def generator(params: Dict):
        try:
            cls = load_impl_ext(mlem_abc.abs_name, type_name=type_name)
        except ImportError:
            return
        yield from _options_from_cls(cls, params)

    return generator


def lazy_class_docstring(abs_name: str, type_name: str):
    def load_docstring():
        try:
            return load_impl_ext(abs_name, type_name).__doc__
        except ExtensionRequirementError as e:
            return f"Help unavailbale: {e}"

    return load_docstring


def for_each_impl(mlem_abc: Type[MlemABC]):
    def inner(f):
        for type_name in list_implementations(mlem_abc):
            f(type_name)
        return f

    return inner


def _iter_errors(
    errors: Sequence[Any], model: Type, loc: Optional[Tuple] = None
):
    for error in errors:
        if isinstance(error, ErrorWrapper):

            if loc:
                error_loc = loc + error.loc_tuple()
            else:
                error_loc = error.loc_tuple()

            if isinstance(error.exc, ValidationError):
                yield from _iter_errors(
                    error.exc.raw_errors, error.exc.model, error_loc
                )
            else:
                yield error_loc, model, error.exc


def _format_validation_error(error: ValidationError) -> List[str]:
    res = []
    for loc, model, exc in _iter_errors(error.raw_errors, error.model):
        path = ".".join(loc_part for loc_part in loc if loc_part != "__root__")
        field_type = model.__fields__[loc[-1]].type_
        if (
            isinstance(exc, MissingError)
            and isinstance(field_type, type)
            and issubclass(field_type, BaseModel)
        ):
            msgs = [
                str(EMOJI_FAIL + f"field `{path}.{f.name}`: {exc}")
                for f in field_type.__fields__.values()
                if f.required
            ]
            if msgs:
                res.extend(msgs)
            else:
                res.append(str(EMOJI_FAIL + f"field `{path}`: {exc}"))
        else:
            res.append(str(EMOJI_FAIL + f"field `{path}`: {exc}"))
    return res


@contextlib.contextmanager
def wrap_build_error(subtype, model: Type[MlemABC]):
    try:
        yield
    except ValidationError as e:
        if LOCAL_CONFIG.DEBUG:
            raise
        msgs = "\n".join(_format_validation_error(e))
        raise typer.BadParameter(
            f"Error on constructing {subtype} {model.abs_name}:\n{msgs}"
        ) from e


def config_arg(
    model: Type[MlemABC],
    load: Optional[str],
    subtype: str,
    conf: Optional[List[str]],
    file_conf: Optional[List[str]],
    **kwargs,
):
    obj: MlemABC
    if load is not None:
        if issubclass(model, MlemObject):
            obj = load_meta(load, force_type=model)
        else:
            with open(load, "r", encoding="utf8") as of:
                obj = parse_obj_as(model, safe_load(of))
    else:
        if not subtype:
            raise typer.BadParameter(
                f"Cannot configure {model.abs_name}: either subtype or --load should be provided"
            )
        with wrap_build_error(subtype, model):
            obj = build_mlem_object(model, subtype, conf, file_conf, kwargs)

    return obj


def _extract_examples(
    help_str: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if help_str is None:
        return None, None
    try:
        examples = help_str.index("Examples:")
    except ValueError:
        return None, help_str
    return help_str[examples + len("Examples:") + 1 :], help_str[:examples]
