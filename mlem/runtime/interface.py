import inspect
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

from pydantic import BaseModel, validator

import mlem
import mlem.version
from mlem.constants import (
    PREDICT_ARG_NAME,
    PREDICT_METHOD_NAME,
    PREDICT_PROBA_METHOD_NAME,
)
from mlem.core.base import MlemABC
from mlem.core.data_type import DataType, UnspecifiedDataType
from mlem.core.errors import MlemError
from mlem.core.metadata import load_meta
from mlem.core.model import Argument, ModelType, Signature
from mlem.core.objects import MlemModel

if TYPE_CHECKING:
    from mlem.runtime.server import Server


class ExecutionError(MlemError):
    pass


class InterfaceDescriptor(BaseModel):
    methods: Dict[str, Signature] = {}
    """interface methods"""


class VersionedInterfaceDescriptor(InterfaceDescriptor):
    version: str = mlem.version.__version__
    """mlem version"""


class Interface(ABC, MlemABC):
    """Base class for runtime interfaces.
    Describes a set of methods togerher with their signatures (arguments
    and return type) and executors - actual python callables to be run
    when the method is invoked. Used to setup `Server`"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "interface"

    @abstractmethod
    def get_method_executor(self, method_name: str):
        raise NotImplementedError

    @abstractmethod
    def get_method_names(self) -> List[str]:
        """
        Lists names of methods exposed by interface

        :return: list of names
        """

        raise NotImplementedError

    def execute(self, method: str, args: Dict[str, object]):
        """
        Executes given method with given arguments

        :param method: method name to execute
        :param args: arguments to pass into method
        :return: method result
        """

        self._validate_args(method, args)
        return self.get_method_executor(method)(**args)

    @abstractmethod
    def load(self, uri: str):
        raise NotImplementedError

    def iter_methods(self) -> Iterator[Tuple[str, Signature]]:
        for method in self.get_method_names():
            yield method, self.get_method_signature(method)

    def _validate_args(self, method: str, args: Dict[str, Any]):
        needed_args = self.get_method_args(method)
        missing_args = [arg for arg in needed_args if arg not in args]
        if missing_args:
            raise ExecutionError(
                f"{self} method {method} missing args {', '.join(missing_args)}"
            )

    @abstractmethod
    def get_method_signature(self, method_name: str) -> Signature:
        """
        Gets signature of given method

        :param method_name: name of method to get signature for
        :return: signature
        """
        raise NotImplementedError

    def get_method_docs(self, method_name: str) -> str:
        """Gets docstring for given method

        :param method_name: name of the method
        :return: docstring
        """
        return getattr(self.get_method_executor(method_name), "__doc__", "")

    def get_method_args(self, method_name: str) -> Dict[str, Argument]:
        """
        Gets argument types of given method

        :param method_name: name of method to get argument types for
        :return: list of argument types
        """
        return {
            a.name: a.type_
            for a in self.get_method_signature(method_name).args
        }

    def get_method_returns(self, method_name: str) -> DataType:
        """
        Gets return type of given method

        :param method_name: name of method to get return type for
        :return: return type
        """

        return self.get_method_signature(method_name).returns

    def get_descriptor(self) -> InterfaceDescriptor:
        return VersionedInterfaceDescriptor(
            version=mlem.__version__,
            methods={
                name: self.get_method_signature(name)
                for name in self.get_method_names()
            },
        )


def expose(f):
    f.is_exposed = True
    return f


class SimpleInterface(Interface):
    """Interface that exposes its own methods that marked with `expose`
    decorator"""

    type: ClassVar[str] = "simple"
    methods: InterfaceDescriptor = InterfaceDescriptor()
    """Interface version and methods"""

    def __init__(self, **data: Any):
        methods = {}
        for name in dir(self):
            attr = getattr(self, name, None)
            if getattr(attr, "is_exposed", False):
                methods[name] = Signature(
                    name=name,
                    args=[
                        Argument(name=a, type_=attr.__annotations__[a])
                        for a in inspect.getfullargspec(attr).args[1:]
                    ],
                    returns=attr.__annotations__.get("return"),
                )
        data["methods"] = InterfaceDescriptor(methods=methods)
        super().__init__(**data)

    def get_method_executor(self, method_name: str):
        return getattr(self, method_name)

    def get_method_names(self) -> List[str]:
        return list(self.methods.methods.keys())

    def load(self, uri: str):
        pass

    def get_method_signature(self, method_name: str) -> Signature:
        return self.methods.methods[method_name]


class ModelInterface(Interface):
    """Interface that descibes model methods"""

    class Config:
        exclude = {"model_type"}

    type: ClassVar[str] = "model"
    model_type: ModelType
    """Model metadata"""

    def load(self, uri: str):
        meta = load_meta(uri)
        if not isinstance(meta, MlemModel):
            raise ValueError("Interface can be created only from models")
        if meta.artifacts is None:
            raise ValueError("Cannot load not saved object")
        self.model_type = meta.model_type
        self.model_type.load(meta.artifacts)

    @validator("model_type")
    @classmethod
    def ensure_signature(cls, value: ModelType):
        if any(
            isinstance(a, UnspecifiedDataType)
            for method in value.methods.values()
            for a in method.args
        ) or any(
            isinstance(method.returns, UnspecifiedDataType)
            for method in value.methods.values()
        ):
            raise MlemError(
                "Cannot create interface from model with unspecified signature. Please re-save it and provide `sample_data` argument"
            )
        return value

    @classmethod
    def from_model(cls, model: MlemModel):
        return cls(model_type=model.model_type)

    def get_method_signature(self, method_name: str) -> Signature:
        return self.model_type.methods[method_name]

    def get_method_executor(self, method_name: str):
        signature = self.get_method_signature(method_name)

        def executor(**kwargs):
            args = [
                kwargs[arg.name]
                if arg.required
                else kwargs.get(arg.name, arg.default)
                for arg in signature.args
            ]
            return self.model_type.call_method(method_name, *args)

        return executor

    def get_method_names(self) -> List[str]:
        return list(self.model_type.methods.keys())

    def get_method_docs(self, method_name: str) -> str:
        return getattr(
            self.model_type.model, self.model_type.methods[method_name].name
        ).__doc__

    def get_method_args(self, method_name: str) -> Dict[str, Argument]:
        return {
            a.name: a.type_ for a in self.model_type.methods[method_name].args
        }


def prepare_model_interface(model: MlemModel, server: "Server"):
    interface: Interface = ModelInterface(model_type=model.model_type)
    if server.interface:
        interface = conform_interface(
            server.interface, interface, server.strict_interface
        )
    elif server.standardize:
        interface = conform_interface(
            standard_interface(model.model_type), interface, strict=False
        )
    return interface


class ConformedInterface(Interface):
    type: ClassVar = "_conformed"

    interface: Interface
    descriptor: InterfaceDescriptor
    method_mapping: Dict[str, str]
    args_mapping: Dict[str, Dict[str, str]]

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

    def get_method_signature(self, method_name: str) -> Signature:
        return self.descriptor.methods[method_name]

    @classmethod
    def from_interface(
        cls,
        interface: Interface,
        descriptor: InterfaceDescriptor,
        interface_descriptor: InterfaceDescriptor = None,
    ):
        interface_descriptor = (
            interface_descriptor or interface.get_descriptor()
        )
        method_mapping = {}
        args_mapping = {}
        for name, method in descriptor.methods.items():
            mapped = cls._find_conformed_method(interface_descriptor, method)
            if mapped:
                method_mapping[name] = mapped[0]
                args_mapping[name] = mapped[1]
        if len(method_mapping) == 0:
            raise MlemError(
                f"No methods in {interface} conformed to descriptor"
            )
        return cls(
            interface=interface,
            descriptor=descriptor,
            method_mapping=method_mapping,
            args_mapping=args_mapping,
        )

    @classmethod
    def _find_conformed_method(
        cls, interface_descriptor: InterfaceDescriptor, signature: Signature
    ) -> Optional[Tuple[str, Dict[str, str]]]:
        req_args = [arg for arg in signature.args if arg.required]
        for name, method in interface_descriptor.methods.items():
            if method.returns != signature.returns:
                continue
            arg_mapping = {
                arg.name: newarg.name
                for arg in req_args
                for newarg in method.args
                if newarg.required and arg.type_ == newarg.type_
            }
            if len(arg_mapping) == len(req_args):
                return name, arg_mapping

        return None


def conform_interface(
    descriptor: InterfaceDescriptor, interface: Interface, strict: bool
) -> Interface:
    interface_descriptor = interface.get_descriptor()
    if strict:
        if interface_descriptor.methods != descriptor.methods:
            raise MlemError(
                f"{interface} cannot be conformed to required descriptor"
            )
        return interface

    return ConformedInterface.from_interface(
        interface, descriptor, interface_descriptor
    )


def standard_interface(
    model_type: ModelType,
    predict_method: str = None,
    predict_proba_method: str = None,
) -> InterfaceDescriptor:
    single_arg_methods = [
        name
        for name, method in model_type.methods.items()
        if len([a for a in method.args if a.required]) == 1
    ]
    if len(single_arg_methods) == 0:
        raise MlemError("Cannot create standard interface from model type")
    if predict_proba_method is None:
        if PREDICT_PROBA_METHOD_NAME in model_type.methods:
            predict_proba_method_name = PREDICT_PROBA_METHOD_NAME
        else:
            if len(single_arg_methods) > 1:
                predict_proba_method_name = single_arg_methods[1]
            else:
                predict_proba_method_name = None
    else:
        predict_proba_method_name = predict_proba_method
    if predict_method is None:
        if PREDICT_METHOD_NAME in model_type.methods:
            predict_method_name = PREDICT_METHOD_NAME
        else:
            predict_method_name = [
                a for a in single_arg_methods if a != predict_proba_method_name
            ][0]
    else:
        predict_method_name = predict_method
    predict = model_type.methods[predict_method_name].copy()
    predict.args = [a.copy() for a in predict.args if a.required]
    predict.args[0].name = PREDICT_ARG_NAME
    predict.name = PREDICT_METHOD_NAME
    methods = {PREDICT_METHOD_NAME: predict}

    if predict_proba_method_name:
        predict_proba = model_type.methods[predict_proba_method_name].copy()
        predict_proba.args = [
            a.copy() for a in predict_proba.args if a.required
        ]
        predict_proba.args[0].name = PREDICT_ARG_NAME
        predict_proba.name = PREDICT_PROBA_METHOD_NAME
        methods[PREDICT_PROBA_METHOD_NAME] = predict_proba
    return InterfaceDescriptor(methods=methods)
