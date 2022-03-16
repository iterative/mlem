import ast
import inspect
import io
import logging
import os
import re
import sys
import threading
import warnings
from functools import wraps
from pickle import PickleError
from types import FunctionType, LambdaType, MethodType, ModuleType
from typing import Dict, List, Optional, Set, Union

import dill
import requests
from dill._dill import TypeType, save_type
from isort.deprecated.finders import FindersManager
from isort.settings import Config
from pydantic.main import ModelMetaclass

from mlem.core.requirements import (
    MODULE_PACKAGE_MAPPING,
    CustomRequirement,
    InstallableRequirement,
    Requirements,
)
from mlem.polydantic.core import PolyModelMetaclass
from mlem.utils import importing

logger = logging.getLogger(__name__)

PYTHON_BASE = os.path.dirname(threading.__file__)
#  pylint: disable=no-member,protected-access
IGNORE_TYPES_REQ = (type(Requirements._abc_impl),)  # type: ignore


def analyze_module_imports(module_path):
    module = importing.import_module(module_path)
    requirements = set()
    for _name, obj in module.__dict__.items():
        if isinstance(obj, ModuleType):
            mod = obj
        else:
            mod = get_object_base_module(obj)
        if is_installable_module(mod) and not is_private_module(mod):
            requirements.add(get_module_repr(mod))

    return requirements


def check_pypi_module(
    module_name, module_version=None, raise_on_error=False, warn_on_error=True
):
    """
    Checks that module with given name and (optionally) version exists in PyPi repository.

    :param module_name: name of module to look for in PyPi
    :param module_version: (optional) version of module to look for in PyPi
    :param raise_on_error: raise `ValueError` if module is not found in PyPi instead of returning `False`
    :param warn_on_error: print a warning if module is not found in PyPi
    :return: `True` if module found in PyPi, `False` otherwise
    """
    r = requests.get(f"https://pypi.org/pypi/{module_name}/json")
    if r.status_code != 200:
        msg = f"Cant find package {module_name} in PyPi"
        if raise_on_error:
            raise ValueError(msg)
        if warn_on_error:
            warnings.warn(msg)
        return False
    if (
        module_version is not None
        and module_version not in r.json()["releases"]
    ):
        msg = f"Cant find package version {module_name}=={module_version} in PyPi"
        if raise_on_error:
            raise ImportError(msg)
        if warn_on_error:
            warnings.warn(msg)
        return False
    return True


def get_object_base_module(obj: object) -> Optional[ModuleType]:
    """
    Determines base module of module given object comes from.

    >>> import numpy
    >>> get_object_base_module(numpy.random.Generator)
    <module 'numpy' from '...'>

    Essentially this function is a combination of :func:`get_object_module` and :func:`get_base_module`.

    :param obj: object to determine base module for
    :return: Python module object for base module
    """
    mod = inspect.getmodule(obj)
    return get_base_module(mod)


def get_base_module(mod: Optional[ModuleType]) -> Optional[ModuleType]:
    """
    Determines base module for given module.

    >>> import mlem.contrib.numpy
    >>> get_base_module(mlem.contrib.numpy)
    <module 'mlem' from '...'>

    :param mod: Python module object to determine base module for
    :return: Python module object for base module
    """
    if mod is None:
        mod = inspect.getmodule(type(mod))
    if mod is None:
        return None
    base, _sep, _stem = mod.__name__.partition(".")
    return sys.modules[base]


def get_object_module(obj: object) -> Optional[ModuleType]:
    """
    Determines module given object comes from

    >>> import numpy
    >>> get_object_module(numpy.ndarray)
    <module 'numpy' from '...'>

    :param obj: obj to determine module it comes from
    :return: Python module object for object module
    """
    return inspect.getmodule(obj)


mlem_isort_config = Config(
    settings_file=os.path.join(os.path.dirname(__file__), "mlem.isort.cfg")
)
isort_finder = FindersManager(config=mlem_isort_config)


def is_private_module(mod: ModuleType):
    """
    Determines that given module object represents private module.

    :param mod: module object to use
    :return: boolean flag
    """
    return mod.__name__.startswith("_")


def is_pseudo_module(mod: ModuleType):
    """
    Determines that given module object represents pseudo (aka Python "magic") module.

    :param mod: module object to use
    :return: boolean flag
    """
    return mod.__name__.startswith("__") and mod.__name__.endswith("__")


def is_extension_module(mod: ModuleType):
    """
    Determines that given module object represents native code extension module.

    :param mod: module object to use
    :return: boolean flag
    """
    try:
        path = mod.__file__
        return any(path.endswith(ext) for ext in (".so", ".pyd"))
    except AttributeError:
        return True


def is_installable_module(mod: ModuleType):
    """
    Determines that given module object represents PyPi-installable (aka third party) module.

    :param mod: module object to use
    :return: boolean flag
    """
    return isort_finder.find(mod.__name__) == "THIRDPARTY"


def is_builtin_module(mod: ModuleType):
    """
    Determines that given module object represents standard library (aka builtin) module.

    :param mod: module object to use
    :return: boolean flag
    """
    return isort_finder.find(mod.__name__) == "STDLIB"


def is_mlem_module(mod: ModuleType):
    """
    Determines that given module object is mlem module

    :param mod: module object to use
    :return: boolean flag
    """
    return mod.__name__ == "mlem" or mod.__name__.startswith("mlem.")


def is_local_module(mod: ModuleType):
    """
    Determines that given module object represents local module.
    Local module is a module (Python file) which is not from standard library and not installed via pip.

    :param mod: module object to use
    :return: boolean flag
    """
    return (
        not is_pseudo_module(mod)
        and not is_mlem_module(mod)
        and not is_builtin_module(mod)
        and not is_installable_module(mod)
        and not is_extension_module(mod)
    )


def is_from_installable_module(obj: object):
    """
    Determines that given object comes from PyPi-installable (aka third party) module.

    :param obj: object to check
    :return: boolean flag
    """
    mod = get_object_base_module(obj)
    if mod is None:
        return False
    return is_installable_module(mod)


def get_module_version(mod: ModuleType):
    """
    Determines version of given module object.

    :param mod: module object to use
    :return: version as `str` or `None` if version could not be determined
    """
    for attr in "__version__", "VERSION":
        if hasattr(mod, attr):
            return getattr(mod, attr)
    for name in os.listdir(os.path.dirname(mod.__file__)):
        m = re.match(re.escape(mod.__name__) + "-(.+)\\.dist-info", name)
        if m:
            return m.group(1)
    return None


def get_python_version():
    """
    :return: Current python version in 'major.minor.micro' format
    """
    major, minor, micro, *_ = sys.version_info
    return f"{major}.{minor}.{micro}"


def get_package_name(mod: ModuleType) -> str:
    """
    Determines PyPi package name for given module object

    :param mod: module object to use
    :return: name as `str`
    """
    if mod is None:
        raise ValueError("mod must not be None")
    name = mod.__name__
    return MODULE_PACKAGE_MAPPING.get(name, name)


def get_module_repr(mod: ModuleType, validate_pypi=False) -> str:
    """
    Builds PyPi `requirements.txt`-compatible representation of given module object

    :param mod: module object to use
    :param validate_pypi: if `True` (default is `False`) perform representation validation in PyPi repository
    :return: representation as `str`
    """
    if mod is None:
        raise ValueError("mod must not be None")
    mod_name = get_package_name(mod)
    mod_version = get_module_version(mod)
    rpr = f"{mod_name}=={mod_version}"
    if validate_pypi:
        check_pypi_module(mod_name, mod_version, raise_on_error=True)
    return rpr


def get_module_as_requirement(
    mod: ModuleType, validate_pypi=False
) -> InstallableRequirement:
    """
    Builds MLEM representation of given module object

    :param mod: module object to use
    :param validate_pypi: if `True` (default is `False`) perform representation validation in PyPi repository
    :return: representation as :class:`.InstallableRequirement`
    """
    mod_version = get_module_version(mod)
    if validate_pypi:
        mod_name = get_package_name(mod)
        check_pypi_module(mod_name, mod_version, raise_on_error=True)
    return InstallableRequirement(module=mod.__name__, version=mod_version)


def get_local_module_reqs(mod):
    tree = ast.parse(inspect.getsource(mod))
    imports = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            imports += [(n.name, None) for n in statement.names]
        elif isinstance(statement, ast.ImportFrom):
            if statement.level == 0:
                imp = (statement.module, None)
            else:
                imp = ("." + statement.module, mod.__package__)
            imports.append(imp)

    result = [importing.import_module(i, p) for i, p in imports]
    if mod.__file__.endswith("__init__.py"):
        # add loaded subpackages
        prefix = mod.__name__ + "."
        result += [
            mod for name, mod in sys.modules.items() if name.startswith(prefix)
        ]
    return result


def lstrip_lines(lines: Union[str, List[str]], check=True) -> str:
    """Lstrip the same amount of spaces from all lines"""
    if isinstance(lines, str):
        lines = lines.splitlines()
    first = lines[0]
    to_strip = len(first) - len(first.lstrip())
    if check and not all(
        line.startswith(" " * to_strip) or line == "" for line in lines
    ):
        raise IndentationError("\n".join(lines))
    return "\n".join(line[to_strip:] for line in lines)


_SKIP_CLOSURE_OBJECTS: Dict[str, Dict[str, Set[str]]] = {
    "globals": {"re": {"_cache"}},
    "nonlocals": {},
}


def add_closure_inspection(f):
    @wraps(f)
    def wrapper(pickler: "RequirementAnalyzer", obj):
        base_module = get_object_base_module(obj)
        if base_module is not None and is_builtin_module(base_module):
            return f(pickler, obj)
        base_module_name = getattr(base_module, "__name__", None)

        closure = inspect.getclosurevars(obj)
        for field in ["nonlocals", "globals"]:
            for k, o in getattr(closure, field).items():
                if k in _SKIP_CLOSURE_OBJECTS[field].get(
                    base_module_name, set()
                ):
                    continue
                if isinstance(o, ModuleType):
                    pickler.add_requirement(o)
                else:
                    pickler.save(o)

        if is_from_installable_module(obj):
            return f(pickler, obj)

        class ImportFromVisitor(ast.NodeVisitor):
            def visit_ImportFrom(self, node: ast.ImportFrom):  # noqa
                warnings.warn(
                    f"Detected local import in {obj.__module__}.{obj.__name__}"
                )
                if node.level == 0:
                    # TODO: https://github.com/iterative/mlem/issues/33
                    mod = importing.import_module(node.module)  # type: ignore
                else:
                    mod = importing.import_module(
                        "." + node.module, get_object_module(obj).__package__  # type: ignore
                    )
                pickler.add_requirement(mod)

        # to add from local imports inside user (non PIP package) code
        try:
            tree = ast.parse(lstrip_lines(inspect.getsource(obj)))
            ImportFromVisitor().visit(tree)
        except OSError:
            logger.debug(
                "Cannot parse code for %s from %s",
                obj,
                inspect.getfile(obj),
                exc_info=True,
            )
        except Exception as e:
            raise Exception(
                f"Cannot parse code for {obj} from {inspect.getfile(obj)}"
            ) from e

        return f(pickler, obj)

    return wrapper


def save_type_with_classvars(pickler: "RequirementAnalyzer", obj):
    for name, attr in obj.__dict__.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        module__ = getattr(get_object_module(obj), "__name__", None)
        if (
            module__ is not None
            and module__.startswith("mlem")
            and not module__.startswith("mlem.contrib")
        ):
            continue
        pickler.save(attr)
    save_type(pickler, obj)


class RequirementAnalyzer(dill.Pickler):
    ignoring = (
        "dill",
        "mlem",
        "pydantic",
        "tests",  # pytest scans all test modules and all their imports are treated as requirements
    )
    dispatch = dill.Pickler.dispatch.copy()

    add_closure_for = [
        FunctionType,
        MethodType,
        staticmethod,
        classmethod,
        LambdaType,
    ]
    dispatch.update(
        {
            t: add_closure_inspection(dill.Pickler.dispatch[t])
            for t in add_closure_for
        }
    )
    dispatch[TypeType] = save_type_with_classvars
    dispatch[ModelMetaclass] = save_type_with_classvars
    dispatch[PolyModelMetaclass] = save_type_with_classvars

    def __init__(self, *args, **kwargs):
        super().__init__(io.BytesIO(), *args, **kwargs)
        self.framer.write = self.skip_write
        self.write = self.skip_write
        self.memoize = self.skip_write
        self.seen = set()
        self._modules = set()

    @property
    def custom_modules(self):
        return {m for m in self._modules if not is_installable_module(m)}

    def to_requirements(self):
        r = Requirements()

        for mod in self._modules:
            if is_installable_module(mod):
                r.add(get_module_as_requirement(get_base_module(mod)))
            elif is_local_module(mod):
                r.add(CustomRequirement.from_module(mod))
        return r

    def _should_ignore(self, mod: ModuleType):
        return (
            any(mod.__name__.startswith(i) for i in self.ignoring)
            or is_private_module(mod)
            or is_pseudo_module(mod)
            or is_builtin_module(mod)
        )

    def add_requirement(self, obj_or_module):
        if not isinstance(obj_or_module, ModuleType):
            try:
                module = get_object_module(obj_or_module)
            except AttributeError as e:
                # Some internal Tensorflow 2.x object crashes `inspect` module on Python 3.6
                logger.debug(
                    "Skipping dependency analysis for %s because of %s: %s",
                    obj_or_module,
                    type(e).__name__,
                    e,
                )
                return
        else:
            module = obj_or_module

        if module is not None and not self._should_ignore(module):
            self._modules.add(module)
            if is_local_module(module):
                # add imports of this module
                for local_req in get_local_module_reqs(module):
                    if local_req in self._modules:
                        continue
                    self.add_requirement(local_req)
                # add imports of subpackage
                if "." in module.__name__:
                    parent_package_name, _ = module.__name__.rsplit(
                        ".", maxsplit=1
                    )
                    if parent_package_name not in self._modules:
                        parent_package = sys.modules[parent_package_name]
                        self.add_requirement(parent_package)

    def save(self, obj, save_persistent_id=True):
        if id(obj) in self.seen or isinstance(obj, IGNORE_TYPES_REQ):
            return None
        self.seen.add(id(obj))
        self.add_requirement(obj)
        try:
            return super().save(obj, save_persistent_id)
        except (ValueError, TypeError, PickleError) as e:
            # if object cannot be serialized, it's probably a C object and we don't need to go deeper
            logger.debug(
                "Skipping dependency analysis for %s because of %s: %s",
                obj,
                type(e).__name__,
                e,
            )
        return None

    def skip_write(self, *args, **kwargs):
        pass


def get_object_requirements(obj) -> Requirements:
    """
    Analyzes packages required for given object to perform its function.
    This function uses `pickle`/`dill` libraries serialization hooks internally.
    Thus result of this function depend on given object being serializable by `pickle`/`dill` libraries:
    all nodes in objects graph which can't be serialized are skipped and their dependencies are lost.

    :param obj: obj to analyze
    :return: :class:`.Requirements` object containing all required packages
    """
    # there was recurse=True here for some reason, but this caused infinite recursion in some cases
    # (def a(): b(); def b(): a(); def func(): a(); get_obj_reqs(func) - smth like that)
    # I removed it and none of the tests failed ¯\_(ツ)_/¯
    a = RequirementAnalyzer()
    a.dump(obj)
    return a.to_requirements()


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
