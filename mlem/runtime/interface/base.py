import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Iterator, List, Tuple

from pydantic import BaseModel

import mlem
import mlem.version
from mlem.core.base import MlemObject
from mlem.core.dataset_type import DatasetType
from mlem.core.errors import MlemError
from mlem.core.metadata import load_meta
from mlem.core.model import Argument, ModelType, Signature
from mlem.core.objects import ModelMeta


class ExecutionError(MlemError):
    pass


class InterfaceDescriptor(BaseModel):
    version: str = mlem.version.__version__
    methods: Dict[str, Signature] = {}


class Interface(ABC, MlemObject):
    class Config:
        type_root = True

    abs_name: ClassVar[str] = "interface"

    @abstractmethod
    def get_method_executor(self, method_name: str):
        raise NotImplementedError()

    @abstractmethod
    def get_method_names(self) -> List[str]:
        """
        Lists names of methods exposed by interface

        :return: list of names
        """

        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def get_method_docs(self, method_name: str) -> str:
        """Gets docstring for given method

        :param method_name: name of the method
        :return: docstring
        """
        return getattr(self.get_method_executor(method_name), "__doc__", None)

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

    def get_method_returns(self, method_name: str) -> DatasetType:
        """
        Gets return type of given method

        :param method_name: name of method to get return type for
        :return: return type
        """

        return self.get_method_signature(method_name).returns

    def get_descriptor(self) -> InterfaceDescriptor:
        return InterfaceDescriptor(
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
    type: ClassVar[str] = "simple"
    methods: InterfaceDescriptor = InterfaceDescriptor()

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
    class Config:
        exclude = {"model_type"}

    type: ClassVar[str] = "model"
    model_type: ModelType

    def load(self, uri: str):
        meta = load_meta(uri)
        if not isinstance(meta, ModelMeta):
            raise ValueError("Interface can be created only from models")
        if meta.artifacts is None:
            raise ValueError("Cannot load not saved object")
        self.model_type = meta.model_type
        self.model_type.load(meta.artifacts)

    @classmethod
    def from_model(cls, model: ModelMeta):
        return cls(model_type=model.model_type)

    def get_method_signature(self, method_name: str) -> Signature:
        return self.model_type.methods[method_name]

    def get_method_executor(self, method_name: str):
        signature = self.get_method_signature(method_name)

        def executor(**kwargs):
            args = [kwargs[arg.name] for arg in signature.args]
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
