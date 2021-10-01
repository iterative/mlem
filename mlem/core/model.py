import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, TypeVar

from fsspec import AbstractFileSystem
from pydantic import BaseModel

from mlem.core.artifacts import Artifacts
from mlem.core.base import MlemObject
from mlem.core.dataset_type import DatasetType
from mlem.core.hooks import Analyzer, Hook
from mlem.core.requirements import WithRequirements


class ModelIO(MlemObject):
    """
    IO base class for models
    """

    __type_root__ = True
    abs_name: ClassVar = "model_io"

    @abstractmethod
    def dump(self, fs: AbstractFileSystem, path, model) -> Artifacts:
        """ """
        raise NotImplementedError()

    @abstractmethod
    def load(self, fs: AbstractFileSystem, path):
        """
        Must load and return model
        :param path: path to load model from
        :return: model object
        """
        raise NotImplementedError()


class SimplePickleIO(ModelIO):
    file_name: ClassVar[str] = "data.pkl"
    type: ClassVar[str] = "simple_pickle"

    def dump(self, fs: AbstractFileSystem, path: str, model) -> Artifacts:
        fs.makedirs(path, exist_ok=True)
        path = os.path.join(path, self.file_name)
        with fs.open(path, "wb") as f:
            pickle.dump(model, f)
        return [path]

    def load(self, fs: AbstractFileSystem, path: str):
        with fs.open(os.path.join(path, self.file_name), "rb") as f:
            return pickle.load(f)


class Argument(BaseModel):
    key: str
    type: DatasetType


class Signature(BaseModel):
    name: str
    args: List[Argument]
    returns: DatasetType


T = TypeVar("T", bound="ModelType")


class ModelType(ABC, MlemObject, WithRequirements):
    """
    Base class for dataset type metadata.
    """

    __type_root__: ClassVar[bool] = True
    abs_name: ClassVar = "model_type"
    __transient_fields__ = {"model"}
    model: Any = None

    io: ModelIO
    methods: Dict[
        str, Signature
    ]  # TODO: https://github.com/iterative/mlem/issues/21

    def load(self, fs: AbstractFileSystem, path: str):
        self.model = self.io.load(fs, path)

    def dump(self, fs: AbstractFileSystem, path: str):
        return self.io.dump(fs, path, self.model)

    def bind(self: T, model: Any) -> "T":
        self.model = model
        return self

    def call_method(self, name, *input_data):
        """
        Calls model method with given name on given input data

        :param name: name of the method to call
        :param input_data: argument for the method
        :return: call result
        """
        self._check_method(name)
        signature = self.methods[name]
        output_data = self._call_method(signature.name, *input_data)
        return output_data

    def _check_method(self, name):
        if self.model is None:
            raise ValueError(f"Model {self} is not loaded")
        if name not in self.methods:
            raise ValueError(f"Model '{self}' doesn't expose method '{name}'")

    def _call_method(self, wrapped: str, *input_data):
        # with switch_curdir(self.curdir):
        if hasattr(self, wrapped):
            return getattr(self, wrapped)(*input_data)
        return getattr(self.model, wrapped)(*input_data)

    def resolve_method(self, method_name: str = None):
        """Checks if method with this name exists

        :param method_name: name of the method.
        If not provided, this model must have only one method and it will be used"""
        if method_name is None:
            if len(self.methods) > 1:
                raise ValueError(
                    f"Please provide one of {list(self.methods.keys())} as method name"
                )
            method_name = next(iter(self.methods))
        self._check_method(method_name)
        return method_name

    def unbind(self):
        self.model = None
        return self


class ModelHook(Hook[ModelType], ABC):
    pass


class ModelAnalyzer(Analyzer[ModelType]):
    base_hook_class = ModelHook
