"""
Base classes to work with requirements which come with ML models and datasets
"""
import base64
import contextlib
import glob
import itertools
import json
import os
import sys
import tempfile
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from types import ModuleType
from typing import (
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from mlem.core.base import MlemObject

# I dont know how to do this better
from mlem.core.errors import HookNotFound
from mlem.core.hooks import Analyzer, Hook
from mlem.utils.importing import import_module

MODULE_PACKAGE_MAPPING = {
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
}
PACKAGE_MODULE_MAPPING = {v: k for k, v in MODULE_PACKAGE_MAPPING.items()}


class Requirement(MlemObject):
    """
    Base class for python requirement
    """

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "requirement"
    type: ClassVar = ...


class PythonRequirement(Requirement, ABC):
    module: str


class InstallableRequirement(PythonRequirement):
    """
    This class represents pip-installable python library

    :param module: name of python module
    :param version: version of python package
    :param package_name: Optional. pip package name for this module, if it is different from module name
    """

    type: ClassVar[str] = "installable"

    module: str
    version: Optional[str] = None
    package_name: Optional[str] = None

    @property
    def package(self):
        """
        Pip package name
        """
        return self.package_name or MODULE_PACKAGE_MAPPING.get(
            self.module, self.module
        )

    def to_str(self):
        """
        pip installable representation of this module
        """
        if self.version is not None:
            return f"{self.package}=={self.version}"
        return self.package

    @classmethod
    def from_module(
        cls, mod: ModuleType, package_name: str = None
    ) -> "InstallableRequirement":
        """
        Factory method to create :class:`InstallableRequirement` from module object

        :param mod: module object
        :param package_name: PIP package name if it is not equal to module name
        :return: :class:`InstallableRequirement`
        """
        from mlem.utils.module import get_module_version

        return InstallableRequirement(
            module=mod.__name__,
            version=get_module_version(mod),
            package_name=package_name,
        )

    @classmethod
    def from_str(cls, name):
        """
        Factory method for creating :class:`InstallableRequirement` from string

        :param name: string representation
        :return: :class:`InstallableRequirement`
        """
        for rel in [
            "==",
            ">=",
            "<=",
        ]:  # TODO for now we interpret everything as exact version
            # https://github.com/iterative/mlem/issues/49
            if rel in name:
                package, version = name.split(rel)
                return InstallableRequirement(module=package, version=version)

        return InstallableRequirement(
            module=name
        )  # FIXME for other relations like > < !=


class CustomRequirement(PythonRequirement):
    """
    This class represents local python code that you need as a requirement for your code

    :param name: filename of this code
    :param source64zip: zipped and base64-encoded source
    :param is_package: whether this code should be in %name%/__init__.py
    """

    type: ClassVar[str] = "custom"
    name: str
    source64zip: str
    is_package: bool

    @staticmethod
    def from_module(mod: ModuleType) -> "CustomRequirement":
        """
        Factory method to create :class:`CustomRequirement` from module object

        :param mod: module object
        :return: :class:`CustomRequirement`
        """
        is_package = mod.__file__.endswith("__init__.py")
        if is_package:
            pkg_dir = os.path.dirname(mod.__file__)
            par = os.path.dirname(pkg_dir)
            sources = {
                os.path.relpath(p, par): Path(p).read_bytes()
                for p in glob.glob(
                    os.path.join(pkg_dir, "**", "*"), recursive=True
                )
                if os.path.isfile(p)
            }
            src = CustomRequirement.compress_package(sources)
        else:
            src = CustomRequirement.compress(
                Path(mod.__file__).read_text(encoding="utf8")
            )
        return CustomRequirement(
            module=mod.__name__.split(".")[0],
            name=mod.__name__,
            source64zip=src,
            is_package=is_package,
        )

    @staticmethod
    def compress(s: str) -> str:
        """
        Helper method to compress source code

        :param s: source code
        :return: base64 encoded string of zipped source
        """
        zp = zlib.compress(s.encode("utf8"))
        b64 = base64.standard_b64encode(zp)
        return b64.decode("utf8")

    @staticmethod
    def compress_package(s: Dict[str, bytes]) -> str:
        sources = {
            path: base64.standard_b64encode(zlib.compress(payload)).decode(
                "utf8"
            )
            for path, payload in s.items()
        }
        return CustomRequirement.compress(json.dumps(sources))

    @staticmethod
    def decompress(s: str) -> str:
        """
        Helper method to decompress source code

        :param s: compressed source code
        :return: decompressed source code
        """
        zp = base64.standard_b64decode(s.encode("utf8"))
        src = zlib.decompress(zp)
        return src.decode("utf8")

    @staticmethod
    def decompress_package(s: str) -> Dict[str, bytes]:
        sources = json.loads(CustomRequirement.decompress(s))
        return {
            path: zlib.decompress(
                base64.standard_b64decode(payload.encode("utf8"))
            )
            for path, payload in sources.items()
        }

    @property
    def module(self):
        """
        Module name for this requirement
        """
        return self.name.split(".")[0]

    @property
    def source(self) -> str:
        """
        Source code of this requirement
        """
        if not self.is_package:
            return CustomRequirement.decompress(self.source64zip)
        raise AttributeError(
            "package requirement does not have source attribute"
        )

    @property
    def sources(self) -> Dict[str, bytes]:
        if self.is_package:
            return CustomRequirement.decompress_package(self.source64zip)
        raise AttributeError(
            "non package requirement does not have sources attribute"
        )

    def to_sources_dict(self) -> Dict[str, bytes]:
        """
        Mapping path -> source code for this requirement

        :return: dict path -> source
        """
        if self.is_package:
            return self.sources
        return {
            self.name.replace(".", "/") + ".py": self.source.encode("utf8")
        }


class FileRequirement(CustomRequirement):
    type: ClassVar[str] = "file"
    is_package: bool = False
    module: str = ""

    def to_sources_dict(self):
        """
        Mapping path -> source code for this requirement

        :return: dict path -> source
        """
        return {self.name: self.source}

    @classmethod
    def from_path(cls, path: str):
        return FileRequirement(
            name=path,
            source64zip=cls.compress(Path(path).read_text(encoding="utf8")),
        )


class UnixPackageRequirement(Requirement):
    type: ClassVar[str] = "unix"
    package_name: str


T = TypeVar("T", bound=Requirement)


class Requirements(BaseModel):
    """
    A collection of requirements

    :param requirements: list of :class:`Requirement` instances
    """

    __root__: List[Requirement] = []

    @property
    def installable(self) -> List[InstallableRequirement]:
        """
        List of installable requirements
        """
        return self.of_type(InstallableRequirement)

    @property
    def custom(self) -> List[CustomRequirement]:
        """
        List of custom requirements
        """
        return self.of_type(CustomRequirement)

    def of_type(self, type_: Type[T]) -> List[T]:
        """
        :param type_: type of requirements
        :return: List of requirements of type `type_`
        """
        return [r for r in self.__root__ if isinstance(r, type_)]

    @property
    def modules(self) -> List[str]:
        """
        List of module names
        """
        return [r.module for r in self.of_type(PythonRequirement)]

    @property
    def expanded(self) -> "Requirements":
        return expand_requirements(self)

    def _add_installable(self, requirement: InstallableRequirement):
        for req in self.installable:
            if req.package == requirement.package:
                if req.version == requirement.version:
                    break
                if (
                    req.version is not None
                    and req.version != requirement.version
                ):
                    raise ValueError(
                        f"Conflicting versions for package {req.package}: {req.version} and {requirement.version}"
                    )
        else:
            self.__root__.append(requirement)

    def _add_custom_package(self, requirement: CustomRequirement):
        for c_req in self.custom:
            if (
                c_req.name.startswith(requirement.name + ".")
                or c_req.name == requirement.name
            ):
                # existing req is subpackage or equals to new req
                self.__root__.remove(c_req)
            if requirement.name.startswith(c_req.name + "."):
                # new req is subpackage of existing
                break
        else:
            self.__root__.append(requirement)

    def _add_custom(self, requirement: CustomRequirement):
        for c_req in self.custom:
            if c_req.is_package and requirement.name.startswith(
                c_req.name + "."
            ):
                # new req is from existing package
                break
            if not c_req.is_package and c_req.name == requirement.name:
                # new req equals to existing
                break
        else:
            self.__root__.append(requirement)

    def add(self, requirement: Requirement):
        """
        Adds requirement to this collection

        :param requirement: :class:`Requirement` instance to add
        """
        if isinstance(requirement, InstallableRequirement):
            self._add_installable(requirement)
        elif isinstance(requirement, CustomRequirement):
            if requirement.is_package:
                self._add_custom_package(requirement)
            else:
                self._add_custom(requirement)
        else:  # TODO better checks here https://github.com/iterative/mlem/issues/49
            if requirement not in self.__root__:
                self.__root__.append(requirement)

    def to_pip(self) -> List[str]:
        """
        :return: list of pip installable packages
        """
        return [r.to_str() for r in self.installable]

    def __add__(self, other: "AnyRequirements"):
        other = resolve_requirements(other)
        res = Requirements(__root__=[])
        for r in itertools.chain(self.__root__, other.__root__):
            res.add(r)
        return res

    def __iadd__(self, other: "AnyRequirements"):
        return self + other

    @staticmethod
    def resolve(reqs: "AnyRequirements") -> "Requirements":
        return resolve_requirements(reqs)

    @classmethod
    def new(cls, requirements: "AnyRequirements" = None):
        if requirements is None:
            return Requirements(__root__=[])
        return resolve_requirements(requirements)

    def materialize_custom(self, path: str):
        for cr in self.custom:
            for part, src in cr.to_sources_dict().items():
                p = os.path.join(path, part)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.write(src)

    @contextlib.contextmanager
    def import_custom(self):
        if not self.custom:
            yield
            return
        with tempfile.TemporaryDirectory(prefix="mlem_custom_reqs") as dirname:
            self.materialize_custom(dirname)
            sys.path.insert(0, dirname)
            for cr in self.custom:
                import_module(cr.module)
            yield
            sys.path.remove(dirname)


def resolve_requirements(other: "AnyRequirements") -> Requirements:
    """
    Helper method to create :class:`Requirements` from any supported source.
    Supported formats: :class:`Requirements`, :class:`Requirement`, list of :class:`Requirement`,
    string representation or list of string representations

    :param other: requirement in supported format
    :return: :class:`Requirements` instance
    """
    if isinstance(other, Requirements):
        return other

    if isinstance(other, list):
        if len(other) == 0:
            return Requirements.new()

        if isinstance(other[0], str):
            return Requirements(
                __root__=[
                    InstallableRequirement.from_str(r) for r in set(other)
                ]
            )

        if isinstance(other[0], Requirement):
            res = Requirements.new()
            for r in other:
                res.add(r)
            return res

        raise TypeError(
            "only other Requirements, Requirement, list of Requirement objects, string "
            "(or list of strings) can be added"
        )
    if isinstance(other, Requirement):
        return Requirements(__root__=[other])

    if isinstance(other, str):
        return Requirements(__root__=[InstallableRequirement.from_str(other)])

    raise TypeError(
        "only other Requirements, Requirement, list of Requirement objects, string "
        "(or list of strings) can be added"
    )


AnyRequirements = Union[
    Requirements, Requirement, Sequence[Requirement], str, Sequence[str]
]


class WithRequirements:
    def get_requirements(self) -> Requirements:
        from mlem.utils.module import get_object_requirements

        return get_object_requirements(self)


class LibRequirementsMixin(WithRequirements):
    """
    :class:`.DatasetType` mixin which provides requirements list consisting of
    PIP packages represented by module objects in `libraries` field.
    """

    libraries: ClassVar[List[ModuleType]]

    def get_requirements(self) -> Requirements:
        return Requirements.new(
            [InstallableRequirement.from_module(lib) for lib in self.libraries]
        )


class RequirementsHook(Hook[Requirements], ABC):
    @classmethod
    @abstractmethod
    def is_object_valid(cls, obj: Requirement) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def process(cls, obj: Requirement, **kwargs) -> Requirements:
        raise NotImplementedError


class AddRequirementHook(RequirementsHook, ABC):
    to_add: AnyRequirements = []

    @classmethod
    def process(cls, obj: Requirement, **kwargs) -> Requirements:
        return resolve_requirements(cls.to_add) + obj


class RequirementsAnalyzer(Analyzer[Requirements]):
    base_hook_class = RequirementsHook


def expand_requirements(requirements: Requirements) -> Requirements:
    res = Requirements.new()
    for req in requirements.__root__:
        try:
            res += RequirementsAnalyzer.analyze(req)
        except HookNotFound:
            res += req
    return res


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
