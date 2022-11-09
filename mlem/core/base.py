import shlex
from collections import defaultdict
from inspect import isabstract
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from pydantic import BaseModel, parse_obj_as
from pydantic.typing import get_args, is_union
from typing_extensions import Literal, get_origin
from yaml import safe_load

from mlem.core.errors import ExtensionRequirementError, UnknownImplementation
from mlem.polydantic import PolyModel
from mlem.utils.importing import import_string
from mlem.utils.path import make_posix


@overload
def load_impl_ext(
    abs_name: str,
    type_name: Optional[str],
    raise_on_missing: Literal[True] = ...,
) -> Type["MlemABC"]:
    ...


@overload
def load_impl_ext(
    abs_name: str,
    type_name: Optional[str],
    raise_on_missing: Literal[False] = ...,
) -> Optional[Type["MlemABC"]]:
    ...


def load_impl_ext(
    abs_name: str, type_name: Optional[str], raise_on_missing: bool = True
) -> Optional[Type["MlemABC"]]:
    """Sometimes, we will not have subclass imported when we deserialize.
    In that case, we first try to import the type_name string
    (because default for PolyModel._get_alias() is module_name.class_name).
    If that fails, we try to find implementation from entrypoints
    """
    from mlem.utils.entrypoints import (  # circular dependencies
        load_entrypoints,
    )

    if abs_name in MlemABC.abs_types:
        abs_class = MlemABC.abs_types[abs_name]
        if type_name in abs_class.__type_map__:
            return abs_class.__type_map__[type_name]

    if type_name is not None and "." in type_name:
        try:
            obj = import_string(type_name)
            if not issubclass(obj, MlemABC):
                raise ValueError(f"{obj} is not subclass of MlemABC")
            return obj
        except ImportError:
            pass

    eps = load_entrypoints()
    for ep in eps.values():
        if ep.abs_name == abs_name and ep.name == type_name:
            try:
                obj = ep.ep.load()
            except ImportError as e:
                from mlem.ext import ExtensionLoader

                ext = ExtensionLoader.builtin_extensions.get(
                    ep.ep.module_name, None
                )
                reqs: List[str]
                if ext is None:
                    reqs = [e.name] if e.name is not None else []
                    extra = None
                else:
                    reqs = ext.reqs_packages
                    extra = ext.extra
                raise ExtensionRequirementError(
                    ep.name or "", reqs, extra
                ) from e
            if not issubclass(obj, MlemABC):
                raise ValueError(f"{obj} is not subclass of MlemABC")
            return obj
    if raise_on_missing:
        raise ValueError(
            f'Unknown implementation of "{abs_name}": {type_name}'
        )
    return None


MT = TypeVar("MT", bound="MlemABC")


class MlemABC(PolyModel):
    """
    Base class for all MLEM Python objects
    that should be serializable and polymorphic
    """

    abs_types: ClassVar[Dict[str, Type["MlemABC"]]] = {}
    abs_name: ClassVar[str]

    @classmethod
    def __resolve_subtype__(cls, type_name: str) -> Type["MlemABC"]:
        """The __type_map__ contains an entry only if the subclass was imported.
        If it is there, we return it.
        If not, we try to load extension using entrypoints registered in setup.py.
        """
        if type_name in cls.__type_map__:
            child_cls = cls.__type_map__[type_name]
        else:
            child_cls = load_impl_ext(cls.abs_name, type_name)
        return child_cls

    def __init_subclass__(cls: Type["MlemABC"]):
        super().__init_subclass__()
        if cls.__is_root__:
            MlemABC.abs_types[cls.abs_name] = cls

    @classmethod
    def non_abstract_subtypes(cls: Type[MT]) -> Dict[str, Type["MT"]]:
        return {
            k: v
            for k, v in cls.__type_map__.items()
            if not isabstract(v)
            and not v.__dict__.get("__abstract__", False)
            or v.__is_root__
            and v is not cls
        }

    @classmethod
    def load_type(cls, type_name: str):
        try:
            return cls.__resolve_subtype__(type_name)
        except ValueError as e:
            raise UnknownImplementation(type_name, cls.abs_name) from e


_not_set = object()


def get_recursively(obj: dict, keys: List[str]):
    if len(keys) == 1:
        return obj[keys[0]]
    key, keys = keys[0], keys[1:]
    return get_recursively(obj[key], keys)


def smart_split(value: str, char: str, maxsplit: int = None):
    SPECIAL = "\0"
    if char != " ":
        value = value.replace(" ", SPECIAL).replace(char, " ")
    res = [
        s.replace(" ", char).replace(SPECIAL, " ")
        for s in shlex.split(value, posix=True)
    ]
    if maxsplit is None:
        return res
    return res[:maxsplit] + [char.join(res[maxsplit:])]


TMO = TypeVar("TMO", bound=MlemABC)


def build_mlem_object(
    model: Type[TMO],
    subtype: str,
    str_conf: List[str] = None,
    file_conf: List[str] = None,
    conf: Dict[str, Any] = None,
    **kwargs,
) -> TMO:
    not_links, links = parse_links(model, str_conf or [])
    if model.__is_root__:
        kwargs[model.__config__.type_field] = subtype
    return build_model(
        model,
        str_conf=not_links,
        file_conf=file_conf,
        conf=conf,
        **kwargs,
        **links,
    )


def parse_links(model: Type["BaseModel"], str_conf: List[str]):
    from mlem.core.objects import MlemLink, MlemObject

    not_links = []
    links = {}
    link_field_names = [
        name
        for name, f in model.__fields__.items()
        if f.type_ is MlemLink and f.name.endswith("_link")
    ]
    link_mapping = {f[: -len("_link")]: f for f in link_field_names}
    link_mapping = {
        k: v for k, v in link_mapping.items() if k in model.__fields__
    }
    link_types = {
        name: f.type_
        for name, f in model.__fields__.items()
        if name in link_mapping and issubclass(f.type_, MlemObject)
    }
    for c in str_conf:
        keys, value = smart_split(c, "=", 1)
        if keys in link_mapping:
            links[link_mapping[keys]] = MlemLink(
                path=value, link_type=link_types[keys].object_type
            )
        else:
            not_links.append(c)
    return not_links, links


IntStr = Union[int, str]
Keys = Tuple[IntStr, ...]
KeyValue = Tuple[IntStr, Any]
Aggregates = Dict[Keys, List[KeyValue]]
TypeHints = Dict[Keys, Type]


class SmartSplitDict(dict):
    def __init__(
        self,
        value=None,
        sep=".",
        type_field="type",
        model: Type[BaseModel] = None,
    ):
        self.type_field = type_field
        self.sep = sep
        self.type_hints: TypeHints = (
            {} if model is None else self._prepare_type_hints(model)
        )
        super().__init__(value or ())

    @classmethod
    def _prepare_type_hints(cls, model: Type[BaseModel]) -> TypeHints:
        # this works only for first-level fields rn.
        # So f: Dict[int, Dict[int, int]] will fail because
        # Nested dict won't get type hint and be treated as list
        # Also, __root__ element type is not checked
        res: TypeHints = {(): dict}
        for name, field in model.__fields__.items():
            field_type = field.outer_type_
            if not isinstance(field_type, type):
                # Handle generics. Probably will break in complex cases
                origin = get_origin(field_type)
                while is_union(origin):
                    # get first type for union
                    generic_args = get_args(field_type)
                    field_type = generic_args[0]
                    origin = (
                        get_origin(field_type)
                        if not isinstance(field_type, type)
                        else None
                    )
                if origin is not None:
                    res[(name,)] = origin
                    continue

            if issubclass(field_type, BaseModel):
                for prefix, type_hint in cls._prepare_type_hints(
                    field_type
                ).items():
                    if not prefix:
                        continue
                    res[(name,) + prefix] = type_hint

        return res

    def update(self, __m: Dict[Any, Any], **kwargs) -> None:  # type: ignore[override]
        for k, v in __m.items():
            self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, key, value):
        if isinstance(key, str):
            key = tuple(smart_split(key, self.sep))

        for keys, val in self._disassemble(value, key):
            super().__setitem__(keys, val)

    def _disassemble(self, value: Any, key_prefix):
        if isinstance(value, list):
            for i, v in enumerate(value):
                yield from self._disassemble(v, key_prefix + (i,))
            return
        if isinstance(value, dict):
            for k, v in value.items():
                yield from self._disassemble(v, key_prefix + (k,))
            return
        yield key_prefix, value

    def build(self) -> Dict[str, Any]:
        prefix_values: Aggregates = self._aggregate_by_prefix()
        while prefix_values:
            if len(prefix_values) == 1 and () in prefix_values:
                return self._merge_aggregates(prefix_values[()], None)
            max_len = max(len(k) for k in prefix_values)
            to_aggregate: Dict[Keys, Any] = {}
            postponed: Aggregates = defaultdict(list)
            for prefix, values in prefix_values.items():
                if len(prefix) == max_len:
                    to_aggregate[prefix] = self._merge_aggregates(
                        values, self.type_hints.get(prefix)
                    )
                    continue
                postponed[prefix] = values
            aggregated: Aggregates = self._aggregate_by_prefix(to_aggregate)
            for prefix in set(postponed).union(aggregated):
                postponed[prefix].extend(aggregated.get(prefix, []))
            if postponed == prefix_values:
                raise RuntimeError("infinite loop on smartdict builing")
            prefix_values = postponed
        # this can only be reached if loop was not entered
        return {}

    def _merge_aggregates(
        self, values: List[KeyValue], type_hint: Optional[Type]
    ) -> Any:
        if (
            type_hint is list
            or all(isinstance(k, int) for k, _ in values)
            and type_hint is None
        ):
            return self._merge_as_list(values)
        return self._merge_as_dict(values)

    def _merge_as_list(self, values: List[KeyValue]):
        assert all(isinstance(k, int) for k, _ in values)
        index_values = defaultdict(list)
        for index, value in values:
            index_values[index].append(value)
        res = [_not_set] * (int(max(k for k, _ in values)) + 1)
        for i, v in index_values.items():
            res[i] = self._merge_values(v)  # type: ignore[index]
        return res

    def _merge_as_dict(self, values: List[KeyValue]) -> Dict[Any, Any]:
        key_values = defaultdict(list)
        for key, value in values:
            key_values[key].append(value)
        return {k: self._merge_values(v) for k, v in key_values.items()}

    def _merge_values(self, values: List[Any]) -> Any:
        if len(values) == 1:
            return values[0]
        merged = {}
        for value in values:
            if isinstance(value, dict):
                merged.update(value)
            elif isinstance(value, str):
                merged[self.type_field] = value
            else:
                raise ValueError(f"Cannot merge {value.__class__} into dict")
        return merged

    def _aggregate_by_prefix(
        self, values: Dict[Keys, Any] = None
    ) -> Aggregates:
        values = values if values is not None else self
        prefix_values: Aggregates = defaultdict(list)

        for keys, value in values.items():
            prefix, key = keys[:-1], keys[-1]
            if isinstance(key, str) and key.isnumeric():
                key = int(key)
            prefix_values[prefix].append((key, value))
        return prefix_values


def parse_string_conf(conf: List[str]) -> Dict[str, Any]:
    res = SmartSplitDict()
    for c in conf:
        keys, value = smart_split(c, "=")
        res[keys] = value
    return res.build()


TBM = TypeVar("TBM", bound=BaseModel)


def build_model(
    model: Type[TBM],
    str_conf: List[str] = None,
    file_conf: List[str] = None,
    conf: Dict[str, Any] = None,
    **kwargs,
) -> TBM:
    if (
        issubclass(model, MlemABC)
        and model.__is_root__
        and model.__config__.type_field in kwargs
    ):
        type_hint_model: Type[BaseModel] = load_impl_ext(
            model.abs_name, kwargs[model.__config__.type_field]
        )
    else:
        type_hint_model = model
    model_dict = SmartSplitDict(model=type_hint_model)
    model_dict.update(kwargs)
    model_dict.update(conf or {})

    for file in file_conf or []:
        keys, path = smart_split(make_posix(file), "=")
        with open(path, "r", encoding="utf8") as f:
            value = safe_load(f)
        model_dict[keys] = value

    for c in str_conf or []:
        keys, value = smart_split(c, "=", 1)
        if value == "None":
            value = None
        model_dict[keys] = value
    return parse_obj_as(model, model_dict.build())
