from abc import ABC, abstractmethod
from typing import ClassVar, Dict

from fsspec import AbstractFileSystem

from mlem.core.base import MlemObject
from mlem.core.objects import ModelMeta


class EnvDescriptor(MlemObject, ABC):
    __type_root__ = True
    abs_name: ClassVar = "env_descriptor"

    @abstractmethod
    def write_files(self, path: str, fs: AbstractFileSystem):
        raise NotImplementedError()

    def get_env_vars(self) -> Dict[str, str]:
        return {}


class Packager(MlemObject):
    __type_root__: ClassVar[bool] = True
    abs_name: ClassVar = "packager"

    @abstractmethod
    def package(
        self, obj: ModelMeta, out: str
    ):  # todo maybe we can also pack datsets?
        raise NotImplementedError()
