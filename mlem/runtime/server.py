import os
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from mlem.core.base import MlemABC
from mlem.core.data_type import DataType, DataTypeSerializer, Serializer
from mlem.core.errors import MlemError
from mlem.core.requirements import Requirements, WithRequirements
from mlem.runtime.interface import (
    Interface,
    InterfaceArgument,
    InterfaceDataType,
    InterfaceDescriptor,
    InterfaceMethod,
)
from mlem.runtime.middleware import Middlewares
from mlem.utils.module import get_object_requirements

MethodMapping = Dict[str, str]
ArgsMapping = Dict[str, str]
MethodArgsMapping = Dict[str, ArgsMapping]


class ServerDataType(BaseModel):
    data_type: Optional[DataType] = None
    """Change data type"""
    ser: Optional[Serializer] = None
    """Change serializer"""

    def matches_types(self, expected: InterfaceDataType):
        if self.data_type is not None and self.data_type != expected.data_type:
            return False
        return True


class ServerArgument(ServerDataType):
    name: Optional[str] = None
    """If set, match only argument with this name"""

    def find_match(self, options: List[InterfaceArgument]) -> Optional[str]:
        for arg in options:
            if self.matches(arg):
                return arg.name
        return None

    def matches(self, expected: InterfaceArgument):
        if self.name is not None and self.name != expected.name:
            return False
        return self.matches_types(expected)


class ServerMethod(BaseModel):
    name: Optional[str] = None
    """If set, match only method with this name"""
    returns: Optional[ServerDataType] = None
    """Change return signature"""
    args: Dict[str, ServerArgument] = {}
    """Change arguments options"""

    def matches(self, expected: InterfaceMethod) -> Tuple[bool, ArgsMapping]:
        if self.name is not None and self.name != expected.name:
            return False, {}

        if self.returns is not None and not self.returns.matches_types(
            expected.returns
        ):
            return False, {}

        arg_mapping: ArgsMapping = {}

        for name, arg in self.args.items():
            match = arg.find_match(
                [
                    a
                    for a in expected.args
                    if a.name not in arg_mapping.values()
                ]
            )
            if not match:
                return False, {}
            arg_mapping[name] = match

        return True, arg_mapping


ServerMethods = Dict[str, ServerMethod]


class _ServerOptions(BaseModel):
    request_serializer: Optional[Serializer] = None
    """Serializer to use for all requests"""
    response_serializer: Optional[Serializer] = None
    """Serializer to use for all responses"""

    standardize: bool = True
    """Use standard model interface"""
    methods: Optional[ServerMethods] = None
    """Optional augmented interface"""


def standard_methods() -> ServerMethods:
    return {
        "predict": ServerMethod(args={"data": ServerArgument()}),
        "predict_proba": ServerMethod(args={"data": ServerArgument()}),
    }


class Server(MlemABC, ABC, WithRequirements, _ServerOptions):
    """Base class for defining serving logic. Server's serve method accepts
    an instance of `Interface` and should expose all of it's methods via some
    protocol"""

    class Config:
        type_root = True

    type: ClassVar[str]
    abs_name: ClassVar[str] = "server"
    env_vars: ClassVar[Optional[Dict[str, str]]] = None
    additional_source_files: ClassVar[Optional[List[str]]] = None
    port_field: ClassVar[Optional[str]] = None

    middlewares: Middlewares = Middlewares()
    """Middlewares to add to server"""

    # @validator("interface")
    # @classmethod
    # def validate_interface(cls, value):
    #     if not value:
    #         return None
    #     return value

    def start(self, interface: Interface):
        return self.serve(ServerInterface.create(self, interface))

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

    @classmethod
    def _get_serializers(
        cls, signature: InterfaceMethod
    ) -> Tuple[Dict[str, DataTypeSerializer], DataTypeSerializer]:
        arg_serializers = {
            arg.name: arg.get_serializer() for arg in signature.args
        }
        returns = signature.returns.get_serializer()
        return arg_serializers, returns

    def get_requirements(self) -> Requirements:
        return (
            super().get_requirements()
            + get_object_requirements(
                [
                    self.request_serializer,
                    self.response_serializer,
                    self.methods,
                ]
            )
            + self.middlewares.get_requirements()
        )

    def get_ports(self) -> List[int]:
        if self.port_field is not None:
            return [getattr(self, self.port_field)]
        return []


class ServerInterface(Interface):
    type: ClassVar = "_server"

    options: _ServerOptions
    interface: Interface
    method_mapping: MethodMapping
    args_mapping: MethodArgsMapping

    @classmethod
    def create(cls, server: Server, interface: Interface):
        if server.methods is not None:
            methods, args = cls.automatch(server.methods, interface)
        elif server.standardize:
            methods, args = cls.automatch(standard_methods(), interface)
        else:
            methods = {k: k for k in interface.get_method_names()}
            args = {
                m: {
                    arg.name: arg.name
                    for arg in interface.get_method_signature(m).args
                }
                for m in methods
            }

        return cls(
            options=server,
            interface=interface,
            method_mapping=methods,
            args_mapping=args,
        )

    @classmethod
    def automatch(
        cls, methods: ServerMethods, interface: Interface
    ) -> Tuple[MethodMapping, MethodArgsMapping]:
        actual: InterfaceDescriptor = interface.get_descriptor()
        used: Set[str] = set()
        method_mapping = {}
        args_mapping = {}
        for name, method in methods.items():
            match = cls._automatch_method(name, actual, method, used)
            if match is None:
                continue
            matched_method, arg_mapping = match
            method_mapping[name] = matched_method
            args_mapping[name] = arg_mapping
            used.add(matched_method)
        if len(method_mapping) == 0:
            raise MlemError(
                "Cannot match any server method to any interface method"
            )
        return method_mapping, args_mapping

    @classmethod
    def _automatch_method(
        cls,
        name: str,
        actual: InterfaceDescriptor,
        expected: ServerMethod,
        used: Set[str],
    ) -> Optional[Tuple[str, ArgsMapping]]:
        if name in actual.__root__ and name not in used:
            match, arg_mapping = expected.matches(actual.__root__[name])
            if match:
                return name, arg_mapping
        for actual_name, method in actual.__root__.items():
            if actual_name in used:
                continue
            match, arg_mapping = expected.matches(method)
            if match:
                return actual_name, arg_mapping
        return None

    def get_method_executor(self, method_name: str):
        og_executor = self.interface.get_method_executor(
            self.method_mapping[method_name]
        )
        args_map = self.args_mapping[method_name]

        def map_args_executor(*args, **kwargs):
            return og_executor(
                *args, **{args_map[k]: v for k, v in kwargs.items()}
            )

        return map_args_executor

    def get_method_names(self) -> List[str]:
        return list(self.method_mapping.keys())

    def load(self, uri: str):
        raise NotImplementedError

    def _get_response(
        self, method_name: str, signature: InterfaceDataType
    ) -> InterfaceDataType:
        if (
            self.options.methods is None
            or self.options.methods[method_name].returns is None
        ):
            return InterfaceDataType(
                data_type=signature.data_type,
                serializer=(
                    self.options.response_serializer
                    or signature.get_serializer().serializer
                ),
            )

        option: ServerMethod = self.options.methods[method_name]
        assert option.returns is not None  # for mypy
        return InterfaceDataType(
            data_type=(option.returns.data_type or signature.data_type),
            serializer=(
                option.returns.ser
                or self.options.response_serializer
                or signature.get_serializer().serializer
            ),
        )

    def _get_request_arg(
        self,
        method_name: str,
        arg_name: str,
        interface_arguments: List[InterfaceArgument],
    ) -> InterfaceArgument:
        mapped_arg = self.args_mapping[method_name][arg_name]
        signature = [s for s in interface_arguments if s.name == mapped_arg][0]
        if self.options.methods is None:
            return InterfaceArgument(
                name=arg_name,
                data_type=signature.data_type,
                serializer=self.options.request_serializer
                or signature.get_serializer().serializer,
                default=signature.default,
                required=signature.required,
            )
        arg = self.options.methods[method_name].args[arg_name]
        return InterfaceArgument(
            name=arg_name,
            data_type=arg.data_type or signature.data_type,
            serializer=arg.ser
            or self.options.request_serializer
            or signature.get_serializer().serializer,
            default=signature.default,
            required=signature.required,
        )

    def get_method_signature(self, method_name: str) -> InterfaceMethod:
        signature = self.interface.get_method_signature(
            self.method_mapping[method_name]
        )
        return InterfaceMethod(
            name=method_name,
            args=[
                self._get_request_arg(method_name, arg, signature.args)
                for arg in self.args_mapping[method_name]
            ],
            returns=self._get_response(method_name, signature.returns),
        )
