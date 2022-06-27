import shlex
from inspect import isabstract
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, overload

from pydantic import BaseModel, parse_obj_as
from typing_extensions import Literal
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


def set_or_replace(obj: dict, key: str, value: Any, subkey: str = "type"):
    if key in obj:
        old_value = obj[key]
        if (
            isinstance(old_value, str)
            and isinstance(value, dict)
            and subkey not in value
        ):
            value[subkey] = old_value
            obj[key] = value
            return
        if isinstance(old_value, dict) and isinstance(value, str):
            old_value[subkey] = value
            return
    obj[key] = value


def set_recursively(obj: dict, keys: List[str], value: Any):
    if len(keys) == 1:
        set_or_replace(obj, keys[0], value)
        return
    key, keys = keys[0], keys[1:]
    set_or_replace(obj, key, {})
    set_recursively(obj[key], keys, value)


def get_recursively(obj: dict, keys: List[str]):
    if len(keys) == 1:
        return obj[keys[0]]
    key, keys = keys[0], keys[1:]
    return get_recursively(obj[key], keys)


def smart_split(string: str, char: str):
    SPECIAL = "\0"
    if char != " ":
        string = string.replace(" ", SPECIAL).replace(char, " ")
    return [
        s.replace(" ", char).replace(SPECIAL, " ")
        for s in shlex.split(string, posix=True)
    ]


def build_mlem_object(
    model: Type[MlemABC],
    subtype: str,
    str_conf: List[str] = None,
    file_conf: List[str] = None,
    conf: Dict[str, Any] = None,
    **kwargs,
):
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
        keys, value = smart_split(c, "=")
        if keys in link_mapping:
            links[link_mapping[keys]] = MlemLink(
                path=value, link_type=link_types[keys].object_type
            )
        else:
            not_links.append(c)
    return not_links, links


def parse_string_conf(conf: List[str]) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    for c in conf:
        keys, value = smart_split(c, "=")
        set_recursively(res, smart_split(keys, "."), value)
    return res


def build_model(
    model: Type[BaseModel],
    str_conf: List[str] = None,
    file_conf: List[str] = None,
    conf: Dict[str, Any] = None,
    **kwargs,
):
    model_dict: Dict[str, Any] = {}
    kwargs.update(conf or {})
    model_dict.update()
    for key, c in kwargs.items():
        set_recursively(model_dict, smart_split(key, "."), c)

    for file in file_conf or []:
        keys, path = smart_split(make_posix(file), "=")
        with open(path, "r", encoding="utf8") as f:
            value = safe_load(f)
        set_recursively(model_dict, smart_split(keys, "."), value)

    for c in str_conf or []:
        keys, value = smart_split(c, "=")
        if value == "None":
            value = None
        set_recursively(model_dict, smart_split(keys, "."), value)
    return parse_obj_as(model, model_dict)
