"""
Classes and functions to load and work with out-of-the-box included extensions
as well with the custom ones
"""
import importlib
import logging
import re
import sys
from types import ModuleType
from typing import Callable, Dict, List, Optional, Union

from mlem.config import LOCAL_CONFIG
from mlem.utils.importing import (
    import_module,
    module_importable,
    module_imported,
)

logger = logging.getLogger(__name__)


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
        extra: Optional[str] = "",
    ):
        self.force = force
        self.reqs = reqs
        self.module = module
        self.validator = validator
        self.extra = extra
        if extra == "":
            self.extra = module.split(".")[-1]

    def __str__(self):
        return f"<Extension {self.module}>"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.module == other.module

    def __hash__(self):
        return hash(self.module)

    @property
    def reqs_packages(self):
        from mlem.core.requirements import MODULE_PACKAGE_MAPPING

        return [MODULE_PACKAGE_MAPPING.get(r, r) for r in self.reqs]


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
        Extension("mlem.contrib.onnx", ["onnx"], False),
        Extension("mlem.contrib.tensorflow", ["tensorflow"], False),
        Extension("mlem.contrib.torch", ["torch"], False),
        Extension("mlem.contrib.catboost", ["catboost"], False),
        # Extension('mlem.contrib.aiohttp', ['aiohttp', 'aiohttp_swagger']),
        # Extension('mlem.contrib.flask', ['flask', 'flasgger'], False),
        # Extension('mlem.contrib.imageio', ['imageio']),
        Extension("mlem.contrib.lightgbm", ["lightgbm"], False),
        Extension("mlem.contrib.xgboost", ["xgboost"], False),
        Extension("mlem.contrib.docker", ["docker"], False),
        Extension("mlem.contrib.fastapi", ["fastapi", "uvicorn"], False),
        Extension("mlem.contrib.callable", [], True),
        Extension("mlem.contrib.rabbitmq", ["pika"], False, extra="rmq"),
        Extension("mlem.contrib.github", [], True),
        Extension("mlem.contrib.gitlabfs", [], True),
        Extension("mlem.contrib.bitbucketfs", [], True),
        Extension("mlem.contrib.sagemaker", ["sagemaker", "boto3"], False),
        Extension("mlem.contrib.dvc", ["dvc"], False),
        Extension(
            "mlem.contrib.heroku", ["fastapi", "uvicorn", "docker"], False
        ),
        Extension("mlem.contrib.pip", [], False),
        Extension("mlem.contrib.kubernetes", ["kubernetes", "docker"], False),
        Extension("mlem.contrib.requirements", [], False),
        Extension("mlem.contrib.venv", [], False),
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

        for mod in LOCAL_CONFIG.additional_extensions:
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


def get_ext_type(ext: Union[str, Extension]):
    if isinstance(ext, Extension):
        ext_module = ext.module
    else:
        ext_module = ext

    doc = import_module(ext_module).__doc__ or ""
    search = re.search(r"Extension type: (\w*)", doc)
    if search is None:
        raise ValueError(f"{ext_module} extension doesnt define it's type")
    return search.group(1)


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
