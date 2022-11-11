import os
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Tuple

from pydantic import validator

from mlem.core.base import MlemABC
from mlem.core.data_type import DataTypeSerializer, Serializer
from mlem.core.model import Signature
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
    request_serializer: Optional[Serializer] = None
    """Serializer to use for all requests"""
    response_serializer: Optional[Serializer] = None
    """Serializer to use for all responses"""
    serializers: Dict[str, Dict[str, Serializer]] = {}
    """Serializer mapping (method, arg) -> Serializer.
    Use special arg name `returns` for method response serializer.
    Overrides request_serializer and response_serializer fields"""

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

    def get_serializers(
        self, method_name: str, signature: Signature
    ) -> Tuple[Dict[str, DataTypeSerializer], DataTypeSerializer]:
        method_serializers = self.serializers.get(method_name, {})
        response_serializer = method_serializers.get(
            "returns", self.response_serializer
        )
        if response_serializer is None:
            returns = signature.returns.get_serializer()
        else:
            returns = DataTypeSerializer(
                serializer=response_serializer, data_type=signature.returns
            )

        arg_serializers: Dict[str, DataTypeSerializer] = {}
        for arg in signature.args:
            serializer = method_serializers.get(
                arg.name, self.request_serializer
            )
            if serializer is None:
                arg_serializers[arg.name] = arg.type_.get_serializer()
            else:
                arg_serializers[arg.name] = DataTypeSerializer(
                    serializer=serializer, data_type=arg.type_
                )

        return arg_serializers, returns
