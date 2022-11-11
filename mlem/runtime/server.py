import os
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional

from pydantic import validator

from mlem.core.base import MlemABC
from mlem.core.requirements import WithRequirements
from mlem.runtime.interface import Interface, InterfaceDescriptor


class Server(MlemABC, ABC, WithRequirements):
    """Base class for defining serving logic. Server's serve method accepts
    an instance of `Interface` and should expose all of it's methods via some
    protocol"""

    class Config:
        type_root = True

    type: ClassVar[str]
    abs_name: ClassVar[str] = "server"
    env_vars: ClassVar[Optional[Dict[str, str]]] = None
    additional_source_files: ClassVar[Optional[List[str]]] = None

    interface: Optional[InterfaceDescriptor] = None
    """Optional augmented interface"""
    strict_interface: bool = False
    """Whether to force identical interface"""
    standardize: bool = True
    """Whether to conform model interface to standard ("predict" method with single arg "data")"""

    @validator("interface")
    @classmethod
    def validate_interface(cls, value):
        if not value:
            return None
        return value

    @abstractmethod
    def serve(self, interface: Interface):
        raise NotImplementedError

    def get_env_vars(self) -> Dict[str, str]:
        return self.env_vars or {}

    def get_sources(self) -> Dict[str, bytes]:
        res = {}
        for path in self.additional_source_files or []:
            with open(path, "rb") as f:
                res[os.path.basename(path)] = f.read()
        return res
