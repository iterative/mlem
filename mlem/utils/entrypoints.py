import glob
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from inspect import isabstract
from typing import TYPE_CHECKING, Dict, List, Optional, Type, TypeVar, Union

import entrypoints

from mlem.core.base import MlemABC, load_impl_ext
from mlem.utils.importing import import_module

if TYPE_CHECKING:
    from mlem.core.objects import MlemObject


logger = logging.getLogger(__name__)

MLEM_ENTRY_POINT = "mlem.contrib"
MLEM_CONFIG_ENTRY_POINT = "mlem.config"


@dataclass
class Entrypoint:
    name: Optional[str]
    abs_name: str
    ep: entrypoints.EntryPoint

    @classmethod
    def from_entrypoint(cls, ep: entrypoints.EntryPoint):
        if "." not in ep.name:
            return cls(name=None, ep=ep, abs_name=ep.name)
        abs_name, name = ep.name.split(".", maxsplit=2)
        return cls(name=name, ep=ep, abs_name=abs_name)

    @property
    def entry(self):
        name = f".{self.name}" if self.name else ""
        return f"{self.abs_name}{name} = {self.ep.module_name}:{self.ep.object_name}"


@lru_cache()
def load_entrypoints(domain: str = MLEM_ENTRY_POINT) -> Dict[str, Entrypoint]:
    """Load MLEM entrypoints defined in setup.py
    These entrypoints are used later to find out which extensions to load
    when a particular object requires them upon it's deserialision
    """
    eps = entrypoints.get_group_named(domain)
    return {k: Entrypoint.from_entrypoint(ep) for k, ep in eps.items()}


def list_implementations(
    base_class: Union[str, Type[MlemABC]],
    meta_subtype: Union[str, Type["MlemObject"]] = None,
    include_hidden: bool = True,
) -> List[str]:
    from mlem.core.objects import MlemObject

    if isinstance(base_class, type) and issubclass(base_class, MlemABC):
        abs_name = base_class.abs_name

    if (base_class in ("meta", MlemObject)) and meta_subtype is not None:
        if isinstance(meta_subtype, str):
            base_class = meta_subtype
        else:
            base_class = meta_subtype.object_type
        abs_name = "meta"
    resolved_base_class: Optional[Type[MlemABC]] = None
    if isinstance(base_class, str):
        abs_name = base_class
        try:
            resolved_base_class = MlemABC.abs_types[abs_name]
        except KeyError:
            try:
                resolved_base_class = load_impl_ext(abs_name, None)
            except ValueError:
                pass
    else:
        resolved_base_class = base_class
    eps = {
        e.name
        for e in load_entrypoints().values()
        if e.abs_name == abs_name and e.name is not None
    }
    if resolved_base_class is not None:
        eps.update(resolved_base_class.non_abstract_subtypes())
    return sorted(e for e in eps if include_hidden or not e.startswith("_"))


def list_abstractions(
    include_hidden: bool = True,
) -> List[str]:
    eps = {e.abs_name for e in load_entrypoints().values()}
    eps.update(MlemABC.abs_types)
    return [e for e in eps if include_hidden or not e.startswith("_")]


IT = TypeVar("IT")


def find_implementations(
    base: Type[IT],
    root_module_name: str = MLEM_ENTRY_POINT,
    raise_on_error: bool = False,
) -> Dict[Type[IT], str]:
    """Generates dict with MLEM entrypoints which should appear in setup.py.
    Can be used by plugin developers to check if they populated all existing
    entrypoints in setup.py
    """
    root_module = import_module(root_module_name)
    assert root_module.__file__ is not None
    path = os.path.dirname(root_module.__file__)

    impls: Dict[Type[IT], str] = {}
    for pyfile in glob.glob(os.path.join(path, "**", "*.py"), recursive=True):
        module_name = (
            root_module_name
            + "."
            + os.path.relpath(pyfile, path)[: -len(".py")].replace(os.sep, ".")
        )
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        try:
            module = import_module(module_name)
        except ImportError as e:
            print(
                f"Cannot import module {module_name}: {e.__class__} {e.args}"
            )
            if raise_on_error:
                raise
            continue

        for obj in module.__dict__.values():

            # pylint: disable=too-many-boolean-expressions
            if (
                isinstance(obj, type)
                and obj.__module__ == module.__name__
                and issubclass(obj, base)
                and not isabstract(obj)
            ):
                impls[obj] = f"{obj.__module__}:{obj.__name__}"
    return impls


def find_abc_implementations(
    root_module_name: str = MLEM_ENTRY_POINT, raise_on_error: bool = False
):
    impls = find_implementations(
        MlemABC, root_module_name, raise_on_error=raise_on_error
    )
    return {
        MLEM_ENTRY_POINT: [
            f"{obj.abs_name}.{obj.__get_alias__()} = {name}"
            if not obj.__is_root__ or hasattr(obj, obj.__type_field__())
            else f"{obj.abs_name} = {name}"
            for obj, name in impls.items()
            if hasattr(obj, "abs_name")
        ]
    }
