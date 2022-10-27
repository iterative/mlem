import ast
import contextlib
import copy
import inspect
from dataclasses import dataclass
from enum import Enum, EnumMeta
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Type

import typer
from click import Context, MissingParameter
from pydantic import (
    BaseModel,
    MissingError,
    ValidationError,
    create_model,
    parse_obj_as,
)
from pydantic.error_wrappers import ErrorWrapper
from pydantic.fields import (
    MAPPING_LIKE_SHAPES,
    SHAPE_LIST,
    SHAPE_SEQUENCE,
    SHAPE_SET,
    SHAPE_TUPLE,
    SHAPE_TUPLE_ELLIPSIS,
    ModelField,
)
from pydantic.typing import display_as_type, get_args, is_union
from typer.core import TyperOption
from typing_extensions import get_origin
from yaml import safe_load

from mlem import LOCAL_CONFIG
from mlem.core.base import (
    MlemABC,
    build_mlem_object,
    load_impl_ext,
    smart_split,
)
from mlem.core.errors import ExtensionRequirementError, MlemObjectNotFound
from mlem.core.meta_io import Location
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
    """A descriptor of model field to build cli option"""

    path: str
    """a.dotted.path from schema root"""
    required: bool
    allow_none: bool
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
    """Parses cls source to find all classfields followed by docstring expr"""
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
        if field is not None and isinstance(statement, ast.Expr):
            if isinstance(statement.value, ast.Constant) and isinstance(
                statement.value.value, str
            ):
                res[field] = statement.value.value
            if isinstance(statement.value, ast.Str):
                res[field] = statement.value.s
        field = None
    return res


@lru_cache()
def get_field_help(cls: Type, field_name: str):
    """Parses all class mro to find classfield docstring"""
    for base_cls in cls.mro():
        if base_cls is object:
            continue
        try:
            docsting = get_attribute_docstrings(base_cls).get(field_name)
            if docsting:
                return docsting
        except OSError:
            pass
    return "Field docstring missing"


def _get_type_name_alias(type_):
    if not isinstance(type_, type):
        type_ = get_origin(type_)
    return type_.__name__ if type_ is not None else "any"


def anything(type_):
    """Creates special type that is named as original type or collection type
    It returns original object on creation and is needed for nice typename in cli option help"""
    return type(
        _get_type_name_alias(type_), (), {"__new__": lambda cls, value: value}
    )


def optional(type_):
    """Creates special type that is named as original type or collection type
    It allows use string `None` to indicate None value"""
    return type(
        _get_type_name_alias(type_),
        (),
        {
            "__new__": lambda cls, value: None
            if value == "None"
            else type_(value)
        },
    )


def parse_type_field(
    path: str,
    type_: Type,
    help_: str,
    is_list: bool,
    is_mapping: bool,
    required: bool,
    allow_none: bool,
    default: Any,
    root_cls: Type[BaseModel],
) -> Iterator[CliTypeField]:
    """Recursively creates CliTypeFields from field description"""
    if is_list or is_mapping:
        # collection
        yield CliTypeField(
            required=required,
            allow_none=allow_none,
            path=path,
            type_=type_,
            default=default,
            help=help_,
            is_list=is_list,
            is_mapping=is_mapping,
            mapping_key_type=str,
        )
        return

    if (
        isinstance(type_, type)
        and issubclass(type_, MlemABC)
        and type_.__is_root__
    ):
        # mlem abstraction: substitute default and extend help
        if isinstance(default, type_):
            default = default.__get_alias__()
        yield CliTypeField(
            required=required,
            allow_none=allow_none,
            path=path,
            type_=type_,
            help=f"{help_}. One of {list_implementations(type_, include_hidden=False)}. Run 'mlem types {type_.abs_name} <subtype>' for list of nested fields for each subtype",
            default=default,
            is_list=is_list,
            is_mapping=is_mapping,
            mapping_key_type=str,
        )
        return
    if isinstance(type_, type) and issubclass(type_, BaseModel):
        # BaseModel (including MlemABC non-root classes): reqursively get nested
        yield from iterate_type_fields(type_, path, not required, root_cls)
        return
    # probably primitive field
    yield CliTypeField(
        required=required,
        allow_none=allow_none,
        path=path,
        type_=type_,
        default=default,
        help=help_,
        is_list=is_list,
        is_mapping=is_mapping,
        mapping_key_type=str,
    )


def iterate_type_fields(
    cls: Type[BaseModel],
    path: str = "",
    force_not_req: bool = False,
    root_cls: Type[BaseModel] = None,
) -> Iterator[CliTypeField]:
    """Recursively get CliTypeFields from BaseModel"""
    if cls is root_cls:
        # avoid infinite recursion
        return
    root_cls = root_cls or cls
    field: ModelField
    for name, field in sorted(
        cls.__fields__.items(), key=lambda x: not x[1].required
    ):
        name = field.alias or name
        if issubclass(cls, MlemObject) and name in MlemObject.__fields__:
            # Skip base MlemObject fields
            continue
        if (
            issubclass(cls, MlemABC)
            and name in cls.__config__.exclude
            or field.field_info.exclude
        ):
            # Skip excluded fields
            continue
        if name == "__root__":
            fullname = path
        else:
            fullname = name if not path else f"{path}.{name}"

        field_type = field.type_
        # field.type_ is element type for collections/mappings

        if not isinstance(field_type, type):
            # Handle generics. Probably will break in complex cases
            origin = get_origin(field_type)
            if is_union(origin):
                # get first type for union
                generic_args = get_args(field_type)
                field_type = generic_args[0]
            if origin is list or origin is dict:
                # replace with dynamic __root__: Dict/List model
                field_type = create_model(
                    display_as_type(field_type), __root__=(field_type, ...)
                )
            if field_type is Any:
                field_type = anything(field_type)

        if not isinstance(field_type, type):
            # skip too complicated stuff
            continue

        yield from parse_type_field(
            path=fullname,
            type_=field_type,
            help_=get_field_help(cls, name),
            is_list=field.shape in LIST_LIKE_SHAPES,
            is_mapping=field.shape in MAPPING_LIKE_SHAPES,
            required=not force_not_req and bool(field.required),
            allow_none=field.allow_none,
            default=field.default,
            root_cls=root_cls,
        )


@dataclass
class CallContext:
    params: Dict[str, Any]
    extra_keys: List[str]
    regular_options: List[str]


def _options_from_model(
    cls: Type[BaseModel],
    ctx: CallContext,
    path="",
    force_not_set: bool = False,
) -> Iterator[TyperOption]:
    """Generate additional cli options from model field"""
    for field in iterate_type_fields(cls, path=path):
        path = field.path
        if path in ctx.regular_options:
            # add dot if path shadows existing parameter
            # it will be ignored on model building
            path = f".{path}"

        if field.is_list:
            yield from _options_from_list(path, field, ctx)
            continue
        if field.is_mapping:
            yield from _options_from_mapping(path, field, ctx)
            continue
        if issubclass(field.type_, MlemABC) and field.type_.__is_root__:
            yield from _options_from_mlem_abc(
                ctx, field, path, force_not_set=force_not_set
            )
            continue

        yield _option_from_field(field, path, force_not_set=force_not_set)


def _options_from_mlem_abc(
    ctx: CallContext,
    field: CliTypeField,
    path: str,
    force_not_set: bool = False,
):
    """Generate str option for mlem abc type.
    If param is already set, also generate respective implementation fields"""
    assert issubclass(field.type_, MlemABC) and field.type_.__is_root__
    if (
        path in ctx.params
        and ctx.params[path] != NOT_SET
        and ctx.params[path] is not None
    ):
        yield from _options_from_model(
            load_impl_ext(field.type_.abs_name, ctx.params[path]),
            ctx,
            path,
        )
    yield _option_from_field(
        field, path, override_type=str, force_not_set=force_not_set
    )


def _options_from_mapping(path: str, field: CliTypeField, ctx: CallContext):
    """Generate options for mapping and example element.
    If some keys are already set, also generate options for them"""
    mapping_keys = [
        key[len(path) + 1 :].split(".", maxsplit=1)[0]
        for key in ctx.extra_keys
        if key.startswith(path + ".")
    ]
    for key in mapping_keys:
        yield from _options_from_collection_element(
            f"{path}.{key}", field, ctx
        )

    override_type = Dict[str, field.type_]  # type: ignore[name-defined]
    yield _option_from_field(
        field, path, override_type=override_type, force_not_set=True
    )
    yield from _options_from_collection_element(
        f"{path}.key", field, ctx, force_not_set=True
    )


def _options_from_list(path: str, field: CliTypeField, ctx: CallContext):
    """Generate option for list and example element.
    If some indexes are already set, also generate options for them"""
    index = 0
    next_path = f"{path}.{index}"
    while any(p.startswith(next_path) for p in ctx.params) and any(
        v != NOT_SET for p, v in ctx.params.items() if p.startswith(next_path)
    ):
        yield from _options_from_collection_element(next_path, field, ctx)
        index += 1
        next_path = f"{path}.{index}"

    override_type = List[field.type_]  # type: ignore[name-defined]
    yield _option_from_field(
        field, path, override_type=override_type, force_not_set=True
    )
    yield from _options_from_collection_element(
        f"{path}.{index}", field, ctx, force_not_set=True
    )


def _options_from_collection_element(
    path: str,
    field: CliTypeField,
    ctx: CallContext,
    force_not_set: bool = False,
) -> Iterator[TyperOption]:
    """Generate options for collection/mapping values"""
    if issubclass(field.type_, MlemABC) and field.type_.__is_root__:
        yield from _options_from_mlem_abc(
            ctx, field, path, force_not_set=force_not_set
        )
        return
    if issubclass(field.type_, BaseModel):
        yield from _options_from_model(
            field.type_, ctx, path, force_not_set=force_not_set
        )
        return
    yield _option_from_field(field, path, force_not_set=force_not_set)


NOT_SET = "__NOT_SET__"
FILE_CONF_PARAM_NAME = "file_conf"
LOAD_PARAM_NAME = "load"


class SetViaFileTyperOption(TyperOption):
    def process_value(self, ctx: Context, value: Any) -> Any:
        try:
            return super().process_value(ctx, value)
        except MissingParameter:
            if (
                LOAD_PARAM_NAME in ctx.params
                or FILE_CONF_PARAM_NAME in ctx.params
                and any(
                    smart_split(v, "=", 1)[0] == self.name
                    for v in ctx.params[FILE_CONF_PARAM_NAME]
                )
            ):
                return NOT_SET
            raise


def _option_from_field(
    field: CliTypeField,
    path: str,
    override_type: Type = None,
    force_not_set: bool = False,
) -> TyperOption:
    """Create cli option from field descriptor"""
    type_ = override_type or field.type_
    if force_not_set:
        type_ = anything(type_)
    elif field.allow_none:
        type_ = optional(type_)
    option = SetViaFileTyperOption(
        param_decls=[f"--{path}", path.replace(".", "_")],
        type=type_ if not force_not_set else anything(type_),
        required=field.required and not force_not_set,
        default=field.default
        if not field.is_list and not field.is_mapping and not force_not_set
        else NOT_SET,
        help=field.help,
        show_default=not field.required,
    )
    option.name = path
    return option


def abc_fields_parameters(type_name: str, mlem_abc: Type[MlemABC]):
    """Create a dynamic options generator that adds implementation fields"""

    def generator(ctx: CallContext):
        try:
            cls = load_impl_ext(mlem_abc.abs_name, type_name=type_name)
        except ImportError:
            return
        yield from _options_from_model(cls, ctx)

    return generator


def get_extra_keys(args):
    return [a[2:] for a in args if a.startswith("--")]


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


def make_not_required(option: TyperOption):
    option = copy.deepcopy(option)
    option.required = False
    option.default = None
    return option


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
        field_name = loc[-1]
        if field_name not in model.__fields__:
            res.append(
                f"Unknown field '{field_name}'. Fields available: {', '.join(model.__fields__)}"
            )
            continue
        field_type = model.__fields__[field_name].type_
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
    subtype: Optional[str],
    conf: Optional[List[str]],
    file_conf: Optional[List[str]],
    **kwargs,
):
    if load is not None:
        if issubclass(model, MlemObject):
            try:
                return load_meta(load, force_type=model)
            except MlemObjectNotFound:
                pass
        with Location.resolve(load).open("r", encoding="utf8") as of:
            return parse_obj_as(model, safe_load(of))
    if not subtype:
        raise typer.BadParameter(
            f"Cannot configure {model.abs_name}: either subtype or --load should be provided"
        )
    with wrap_build_error(subtype, model):
        return build_mlem_object(model, subtype, conf, file_conf, kwargs)
