from typing import ClassVar, Optional, Type, overload

from typing_extensions import Literal

from mlem.polydantic import PolyModel
from mlem.utils.importing import import_string


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
    def validate(cls, value):
        """Polymorphic magic goes here"""
        if isinstance(value, cls):
            return value
        value = value.copy()
        type_name = value.pop(
            cls.__config__.type_field, cls.__config__.default_type
        )
        if type_name is None:
            raise ValueError(
                f"Type field was not provided and no default type specified in {cls.__parent__.__name__}"
            )
        child_cls: Type[MlemObject] = cls.resolve_subtype(type_name)
        return child_cls(**value)

    @classmethod
    def resolve_subtype(cls, type_name: str) -> Type["MlemObject"]:
        """The __type_map__ contains an entry only if the subclass was imported.
        If it is there, we return it.
        If not, we try to load extension using entrypoints registered in setup.py.
        """
        if type_name in cls.__type_map__:
            child_cls = cls.__type_map__[type_name]
        else:
            child_cls = load_impl_ext(cls.abs_name, type_name)
        return child_cls
