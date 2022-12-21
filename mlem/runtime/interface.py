import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, validator

import mlem
import mlem.version
from mlem.core.base import MlemABC
from mlem.core.data_type import (
    DataType,
    DataTypeSerializer,
    Serializer,
    UnspecifiedDataType,
)
from mlem.core.errors import MlemError
from mlem.core.metadata import load_meta
from mlem.core.model import Argument, Signature
from mlem.core.objects import MlemModel


class ExecutionError(MlemError):
    pass


class InterfaceDataType(BaseModel):
    data_type: DataType
    serializer: Optional[Serializer]

    def get_serializer(self) -> DataTypeSerializer:
        if self.serializer is None:
            return self.data_type.get_serializer()

        return DataTypeSerializer(
            serializer=self.serializer, data_type=self.data_type
        )


class InterfaceArgument(InterfaceDataType):
    name: str
    required: bool = True
    default: Any = None

    @classmethod
    def from_argument(cls, argument: Argument):
        return cls(
            name=argument.name,
            data_type=argument.type_,
            serializer=None,
            required=argument.required,
            default=argument.default,
        )


class InterfaceMethod(BaseModel):
    name: str
    args: List[InterfaceArgument]
    returns: InterfaceDataType

    @classmethod
    def from_signature(cls, signature: Signature) -> "InterfaceMethod":
        args = [
            InterfaceArgument(
                name=a.name,
                data_type=a.type_,
                serializer=a.type_.get_serializer().serializer,
                required=a.required,
                default=a.default,
            )
            for a in signature.args
        ]
        if not args and signature.varargs and signature.varargs_type:
            args = [
                InterfaceArgument(
                    name=signature.varargs,
                    data_type=signature.varargs_type,
                    serializer=signature.varargs_type.get_serializer().serializer,
                    required=False,
                    default=None,
                )
            ]
        # no varkw support yet
        return InterfaceMethod(
            name=signature.name,
            args=args,
            returns=InterfaceDataType(
                data_type=signature.returns,
                serializer=signature.returns.get_serializer().serializer,
            ),
        )


class InterfaceDescriptor(BaseModel):
    __root__: Dict[str, InterfaceMethod] = {}
    """interface methods"""


class VersionedInterfaceDescriptor(BaseModel):
    methods: InterfaceDescriptor
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

    def iter_methods(self) -> Iterator[Tuple[str, InterfaceMethod]]:
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
    def get_method_signature(self, method_name: str) -> InterfaceMethod:
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

    def get_method_args(
        self, method_name: str
    ) -> Dict[str, InterfaceArgument]:
        """
        Gets argument types of given method

        :param method_name: name of method to get argument types for
        :return: list of argument types
        """
        return {a.name: a for a in self.get_method_signature(method_name).args}

    def get_method_returns(self, method_name: str) -> InterfaceDataType:
        """
        Gets return type of given method

        :param method_name: name of method to get return type for
        :return: return type
        """

        return self.get_method_signature(method_name).returns

    def get_descriptor(self) -> InterfaceDescriptor:
        return InterfaceDescriptor(
            __root__={
                name: self.get_method_signature(name)
                for name in self.get_method_names()
            }
        )

    def get_versioned_descriptor(self) -> VersionedInterfaceDescriptor:
        return VersionedInterfaceDescriptor(
            version=mlem.__version__, methods=self.get_descriptor()
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
                methods[name] = InterfaceMethod(
                    name=name,
                    args=[
                        InterfaceArgument(
                            name=a,
                            data_type=attr.__annotations__[a],
                            serializer=None,
                        )
                        for a in inspect.getfullargspec(attr).args[1:]
                    ],
                    returns=InterfaceDataType(
                        data_type=attr.__annotations__.get("return"),
                        serializer=None,
                    ),
                )
        data["methods"] = InterfaceDescriptor(__root__=methods)
        super().__init__(**data)

    def get_method_executor(self, method_name: str):
        return getattr(self, method_name)

    def get_method_names(self) -> List[str]:
        return list(self.methods.__root__.keys())

    def load(self, uri: str):
        pass

    def get_method_signature(self, method_name: str) -> InterfaceMethod:
        return self.methods.__root__[method_name]


def _check_no_signature(data):
    if isinstance(data, dict):
        if "type" in data and data["type"] == UnspecifiedDataType.type:
            return False
        return all(_check_no_signature(v) for v in data.values())
    if isinstance(data, list):
        return all(_check_no_signature(v) for v in data)
    return True


class ModelInterface(Interface):
    """Interface that descibes model methods"""

    type: ClassVar[str] = "model"
    model: MlemModel
    """Model metadata"""

    def load(self, uri: str):
        meta = load_meta(uri)
        if not isinstance(meta, MlemModel):
            raise ValueError("Interface can be created only from models")
        if meta.artifacts is None:
            raise ValueError("Cannot load not saved object")
        self.model = meta
        self.model.load_value()

    @validator("model")
    @classmethod
    def ensure_signature(cls, value: MlemModel):
        if not _check_no_signature(value.processors_raw):
            raise MlemError(
                "Cannot create interface from model with unspecified signature. Please re-save it and provide `sample_data` argument"
            )
        return value

    @classmethod
    def from_model(cls, model: MlemModel):
        return cls(model=model)

    @property
    def is_single_model(self):
        return self.model.is_single_model

    def get_method_signature(self, method_name: str) -> InterfaceMethod:
        if self.is_single_model:
            signature = self.model.model_type.methods[method_name]
            return InterfaceMethod.from_signature(signature)
        call_order = self.model.call_orders[method_name]
        input_signature = self.model.get_processor(call_order[0][0]).methods[
            call_order[0][1]
        ]
        output_signature = self.model.get_processor(call_order[-1][0]).methods[
            call_order[-1][1]
        ]
        return InterfaceMethod.from_signature(
            Signature(
                name=method_name,
                args=input_signature.args,
                varkw=input_signature.varkw,
                varargs=input_signature.varargs,
                varkw_type=input_signature.varkw_type,
                varargs_type=input_signature.varargs_type,
                returns=output_signature.returns,
            )
        )

    def get_method_executor(self, method_name: str):
        signature = self.get_method_signature(method_name)

        def executor(**kwargs):
            args = {
                arg.name: kwargs[arg.name]
                if arg.required
                else kwargs.get(arg.name, arg.default)
                for arg in signature.args
            }
            return getattr(self.model, method_name)(**args)

        return executor

    def get_method_names(self) -> List[str]:
        return list(self.model.call_orders)

    def get_method_docs(self, method_name: str) -> str:
        return getattr(
            self.model.model_type.model,
            self.model.model_type.methods[method_name].name,
        ).__doc__

    def get_method_args(
        self, method_name: str
    ) -> Dict[str, InterfaceArgument]:
        return {
            a.name: a.type_
            for a in self.model.model_type.methods[method_name].args
        }
