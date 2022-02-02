"""
Classes and functions to load and work with out-of-the-box included extensions
as well with the custom ones
"""
import glob
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from inspect import isabstract
from types import ModuleType
from typing import Callable, Dict, List, Union

import entrypoints

from mlem.config import CONFIG
from mlem.core.base import MlemObject
from mlem.utils.importing import (
    import_module,
    module_importable,
    module_imported,
)

logger = logging.getLogger(__name__)

MLEM_ENTRY_POINT = "mlem.contrib"


class Extension:
    """
    Extension descriptor

    :param module: main extension module
    :param reqs: list of extension dependencies
    :param force: if True, disable lazy loading for this extension
    :param validator: boolean predicate which should evaluate to True for this extension to be loaded
    """

    def __init__(
        self,
        module,
        reqs: List[str],
        force: bool = True,
        validator: Callable[[], bool] = None,
    ):
        self.force = force
        self.reqs = reqs
        self.module = module
        self.validator = validator

    def __str__(self):
        return f"<Extension {self.module}>"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.module == other.module

    def __hash__(self):
        return hash(self.module)


class ExtensionDict(dict):
    """
    :class:`_Extension` container
    """

    def __init__(self, *extensions: Extension):
        super().__init__()
        for e in extensions:
            self[e.module] = e


# def __tensorflow_major_version():
#     import tensorflow as tf
#     return tf.__version__.split('.')[0]
#
#
# is_tf_v1, is_tf_v2 = lambda: __tensorflow_major_version() == '1', lambda: __tensorflow_major_version() == '2'


class ExtensionLoader:
    """
    Class that tracks and loads extensions.

    """

    builtin_extensions: Dict[str, Extension] = ExtensionDict(
        Extension("mlem.contrib.numpy", ["numpy"], False),
        Extension("mlem.contrib.pandas", ["pandas"], False),
        Extension("mlem.contrib.sklearn", ["sklearn"], False),
        # Extension('mlem.contrib.tensorflow', ['tensorflow'], False, is_tf_v1),
        # Extension('mlem.contrib.tensorflow_v2', ['tensorflow'], False, is_tf_v2),
        # Extension('mlem.contrib.torch', ['torch'], False),
        Extension("mlem.contrib.catboost", ["catboost"], False),
        # Extension('mlem.contrib.aiohttp', ['aiohttp', 'aiohttp_swagger']),
        # Extension('mlem.contrib.flask', ['flask', 'flasgger'], False),
        # Extension('mlem.contrib.sqlalchemy', ['sqlalchemy']),
        # Extension('mlem.contrib.s3', ['boto3']),
        # Extension('mlem.contrib.imageio', ['imageio']),
        Extension("mlem.contrib.lightgbm", ["lightgbm"], False),
        Extension("mlem.contrib.xgboost", ["xgboost"], False),
        # Extension("mlem.contrib.docker", ["docker"], False),
        Extension("mlem.contrib.fastapi", ["fastapi", "uvicorn"], False),
        Extension("mlem.contrib.callable", [], True),
    )

    _loaded_extensions: Dict[Extension, ModuleType] = {}

    @classmethod
    def loaded_extensions(cls) -> Dict[Extension, ModuleType]:
        """
        :return: List of loaded extensions
        """
        return cls._loaded_extensions

    @classmethod
    def _setup_import_hook(cls, extensions: List[Extension]):
        """
        Add import hook to sys.meta_path that will load extensions when their dependencies are imported

        :param extensions: list of :class:`.Extension`
        """
        if len(extensions) == 0:
            return

        existing = [
            h
            for h in sys.meta_path
            if isinstance(h, _ImportLoadExtInterceptor)  # type: ignore
            # TODO: https://github.com/iterative/mlem/issues/33
        ]
        if len(existing) == 1:
            hook = existing[0]
            hook.module_to_extension.update(  # type: ignore
                # TODO: https://github.com/iterative/mlem/issues/33
                {req: e for e in extensions for req in e.reqs}
            )
        elif len(existing) > 1:
            raise ValueError("Multiple import hooks, that is impossible")
        else:
            hook = _ImportLoadExtInterceptor(  # type: ignore
                # TODO: https://github.com/iterative/mlem/issues/33
                module_to_extension={
                    req: e for e in extensions for req in e.reqs
                }
            )
            sys.meta_path.insert(0, hook)

    @classmethod
    def load_all(cls, try_lazy=True):
        """
        Load all (builtin and additional) extensions

        :param try_lazy: if `False`, use force load for all builtin extensions
        """
        for_hook = []
        for ext in cls.builtin_extensions.values():
            if not try_lazy or hasattr(sys, "frozen") or ext.force:
                if all(module_importable(r) for r in ext.reqs):
                    cls.load(ext)
            else:
                if all(module_imported(r) for r in ext.reqs):
                    cls.load(ext)
                else:
                    for_hook.append(ext)

        cls._setup_import_hook(for_hook)

        for mod in CONFIG.ADDITIONAL_EXTENSIONS:
            cls.load(mod)

    @classmethod
    def load(cls, extension: Union[str, Extension]):
        """
        Load single extension

        :param extension: str or :class:`.Extension` instance to load
        """
        if isinstance(extension, str):
            extension = Extension(extension, [], force=True)
        if extension not in cls._loaded_extensions and (
            extension.validator is None or extension.validator()
        ):
            if not module_imported(extension.module):
                logger.debug("Importing extension module %s", extension.module)
            cls._loaded_extensions[extension] = import_module(extension.module)


class _ImportLoadExtRegisterer(importlib.abc.PathEntryFinder):
    """A hook that registers all modules that are being imported"""

    def __init__(self):
        self.imported = []

    def find_module(
        self, fullname, path=None
    ):  # pylint: disable=unused-argument
        self.imported.append(fullname)


class _ImportLoadExtInterceptor(  # pylint: disable=abstract-method
    importlib.abc.Loader, importlib.abc.PathEntryFinder
):
    """
    Import hook implementation to load extensions on dependency import

    :param module_to_extension: dict requirement -> :class:`.Extension`
    """

    def __init__(self, module_to_extension: Dict[str, Extension]):
        self.module_to_extension = module_to_extension

    def find_module(
        self, fullname, path=None
    ):  # pylint: disable=unused-argument
        # hijack importing machinery
        return self

    def load_module(self, fullname):
        # change this hook to registering hook
        reg = _ImportLoadExtRegisterer()
        sys.meta_path = [reg] + [x for x in sys.meta_path if x is not self]
        try:
            # fallback to ordinary importing
            module = importlib.import_module(fullname)
        finally:
            # put this hook back
            sys.meta_path = [self] + [x for x in sys.meta_path if x is not reg]

        # check all that was imported and import all extensions that are ready
        for imported in reg.imported:
            if not module_imported(imported):
                continue
            extension = self.module_to_extension.get(imported)
            if extension is None:
                continue

            if all(module_imported(m) for m in extension.reqs):
                ExtensionLoader.load(extension)

        return module


def load_extensions(*exts: str):
    """
    Load extensions

    :param exts: list of extension main modules
    """
    for ext in exts:
        ExtensionLoader.load(ext)


@dataclass
class Entrypoint:
    name: str
    abs_name: str
    ep: entrypoints.EntryPoint

    @classmethod
    def from_entrypoint(cls, ep: entrypoints.EntryPoint):
        abs_name = ep.name.split(".")[0]
        name = ep.name[len(abs_name) + 1 :]
        return cls(name=name, ep=ep, abs_name=abs_name)

    @property
    def entry(self):
        return f"{self.abs_name}.{self.name} = {self.ep.module_name}:{self.ep.object_name}"


@lru_cache()
def load_entrypoints() -> Dict[str, Entrypoint]:
    """Load MLEM entrypoints defined in setup.py
    These entrypoints are used later to find out which extensions to load
    when a particular object requires them upon it's deserialision
    """
    eps = entrypoints.get_group_named(MLEM_ENTRY_POINT)
    return {k: Entrypoint.from_entrypoint(ep) for k, ep in eps.items()}


def find_implementations(root_module_name: str = MLEM_ENTRY_POINT):
    """Generates dict with MLEM entrypoints which should appear in setup.py.
    Can be used by plugin developers to check if they populated all existing
    entrypoints in setup.py
    """
    root_module = import_module(root_module_name)
    path = os.path.dirname(root_module.__file__)

    impls = {}
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
            continue

        for obj in module.__dict__.values():

            # pylint: disable=too-many-boolean-expressions
            if (
                isinstance(obj, type)
                and obj.__module__ == module.__name__
                and issubclass(obj, MlemObject)
                and not obj.__is_root__
                and not isabstract(obj)
                and hasattr(obj, "abs_name")
            ):
                impls[obj] = f"{obj.__module__}:{obj.__name__}"

    return {
        MLEM_ENTRY_POINT: [
            f"{obj.abs_name}.{obj.__get_alias__()} = {name}"
            for obj, name in impls.items()
        ]
    }


# Copyright 2019 Zyfra
# Copyright 2021 Iterative
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
