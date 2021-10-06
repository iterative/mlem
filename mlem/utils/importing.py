import importlib.util
import sys
from importlib import import_module


def import_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ImportError(f"Cannot import spec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    sys.modules[name] = module
    return module


def import_string(path):
    split = path.split(".")
    module_name, object_name = ".".join(split[:-1]), split[-1]
    mod = import_module(module_name)
    try:
        return getattr(mod, object_name)
    except AttributeError as e:
        raise ImportError(
            f"No object {object_name} in module {module_name}"
        ) from e


def module_importable(module_name):
    try:
        import_module(module_name)
        return True
    except ImportError:
        return False


def module_imported(module_name):
    """
    Checks if module already imported

    :param module_name: module name to check
    :return: `True` or `False`
    """
    return sys.modules.get(module_name) is not None


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
