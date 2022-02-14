import shlex
from typing import Any, ClassVar, Dict, List, Optional, Type, overload

from pydantic import BaseModel, parse_obj_as
from typing_extensions import Literal
from yaml import safe_load

from mlem.polydantic import PolyModel
from mlem.utils.importing import import_string
from mlem.utils.path import make_posix


@overload
def load_impl_ext(
    abs_name: str, type_name: str, raise_on_missing: Literal[True] = ...
) -> Type["MlemObject"]:
    ...


@overload
def load_impl_ext(
    abs_name: str, type_name: str, raise_on_missing: Literal[False] = ...
) -> Optional[Type["MlemObject"]]:
    ...


def load_impl_ext(
    abs_name: str, type_name: str, raise_on_missing: bool = True
) -> Optional[Type["MlemObject"]]:
    """Sometimes, we will not have subclass imported when we deserialize.
    In that case, we first try to import the type_name string
    (because default for PolyModel._get_alias() is module_name.class_name).
    If that fails, we try to find implementation from entrypoints
    """
    from mlem.ext import load_entrypoints  # circular dependencies

    if "." in type_name:
        try:
            obj = import_string(type_name)
            if not issubclass(obj, MlemObject):
                raise ValueError(f"{obj} is not subclass of MlemObject")
            return obj
        except ImportError:
            pass

    eps = load_entrypoints()
    for ep in eps.values():
        if ep.abs_name == abs_name and ep.name == type_name:
            obj = ep.ep.load()
            if not issubclass(obj, MlemObject):
                raise ValueError(f"{obj} is not subclass of MlemObject")
            return obj
    if raise_on_missing:
        raise ValueError(
            f'Unknown implementation of "{abs_name}": {type_name}'
        )
    return None


class MlemObject(PolyModel):
    """
    Base class for all MLEM Python objects
    which should be serialized and deserialized
    """

    abs_name: ClassVar[str]

    @classmethod
    def __resolve_subtype__(cls, type_name: str) -> Type["MlemObject"]:
        """The __type_map__ contains an entry only if the subclass was imported.
        If it is there, we return it.
        If not, we try to load extension using entrypoints registered in setup.py.
        """
        if type_name in cls.__type_map__:
            child_cls = cls.__type_map__[type_name]
        else:
            child_cls = load_impl_ext(cls.abs_name, type_name)
        return child_cls


def _set_recursively(obj: dict, keys: List[str], value: Any):
    if len(keys) == 1:
        obj[keys[0]] = value
        return
    key, keys = keys[0], keys[1:]
    if key not in obj:
        obj[key] = {}
    _set_recursively(obj[key], keys, value)


def smart_split(string: str, char: str):
    SPECIAL = "\0"
    if char != " ":
        string = string.replace(" ", SPECIAL).replace(char, " ")
    return [
        s.replace(" ", char).replace(SPECIAL, " ")
        for s in shlex.split(string, posix=True)
    ]


def build_mlem_object(
    model: Type[MlemObject],
    subtype: str,
    str_conf: List[str] = None,
    file_conf: List[str] = None,
    conf: Dict[str, Any] = None,
    **kwargs,
):
    not_links, links = parse_links(model, str_conf or [])
    return build_model(
        model,
        str_conf=not_links,
        file_conf=file_conf,
        conf=conf,
        **{model.__config__.type_field: subtype},
        **kwargs,
        **links,
    )


def parse_links(model: Type["BaseModel"], str_conf: List[str]):
    from mlem.core.objects import MlemLink, MlemMeta

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
        if name in link_mapping and issubclass(f.type_, MlemMeta)
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
        _set_recursively(res, smart_split(keys, "."), value)
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
        _set_recursively(model_dict, smart_split(key, "."), c)

    for file in file_conf or []:
        keys, path = smart_split(make_posix(file), "=")
        with open(path, "r", encoding="utf8") as f:
            value = safe_load(f)
        _set_recursively(model_dict, smart_split(keys, "."), value)

    for c in str_conf or []:
        keys, value = smart_split(c, "=")
        _set_recursively(model_dict, smart_split(keys, "."), value)
    return parse_obj_as(model, model_dict)
