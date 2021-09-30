import os
from types import ModuleType
from typing import Any, ClassVar, List, Optional, Tuple, Type, Union

import numpy as np
from fsspec import AbstractFileSystem
from pydantic import BaseModel, conlist, create_model

from mlem.core.artifacts import Artifacts
from mlem.core.dataset_type import (
    Dataset,
    DatasetHook,
    DatasetReader,
    DatasetSerializer,
    DatasetType,
    DatasetWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.requirements import LibRequirementsMixin


def python_type_from_np_string_repr(string_repr: str) -> type:
    np_type = np_type_from_string(string_repr)
    return python_type_from_np_type(np_type)


def python_type_from_np_type(np_type: Union[Type, np.dtype]):
    if np_type.type == np.object_:
        return object
    value = np_type.type()
    if np_type.type.__module__ == "numpy":
        value = value.item()
    return type(value)


def np_type_from_string(string_repr) -> np.dtype:
    try:
        return np.dtype(string_repr)
    except TypeError:
        raise ValueError("Unknown numpy type {}".format(string_repr))


class NumpyNumberType(LibRequirementsMixin, DatasetType, DatasetHook):
    """
    :class:`.DatasetType` implementation for `numpy.number` objects which
    converts them to built-in Python numbers and vice versa.

    :param dtype: `numpy.number` data type as string
    """

    libraries: ClassVar[List[ModuleType]] = [np]
    type: ClassVar[str] = "number"
    dtype: str

    # def get_spec(self) -> ArgList:
    #     return [Field(None, python_type_from_np_string_repr(self.dtype), False)]

    def deserialize(self, obj: dict) -> Any:
        return self.actual_type(obj)

    def serialize(self, instance: np.number) -> object:  # type: ignore
        self._check_type(instance, np.number, ValueError)
        return instance.item()

    @property
    def actual_type(self):
        return np_type_from_string(self.dtype)

    # def get_writer(self):
    #     return PickleWriter()
    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, np.number)

    @classmethod
    def process(cls, obj: np.number, **kwargs) -> "NumpyNumberType":
        return NumpyNumberType(dtype=obj.dtype.name)

    def get_writer(self):
        raise NotImplementedError()

    def get_model(self):
        raise NotImplementedError()


class NumpyNdarrayType(
    LibRequirementsMixin, DatasetType, DatasetHook, DatasetSerializer
):
    """
    :class:`.DatasetType` implementation for `np.ndarray` objects
    which converts them to built-in Python lists and vice versa.

    :param shape: shape of `numpy.ndarray` objects in dataset
    :param dtype: data type of `numpy.ndarray` objects in dataset
    """

    type: ClassVar[str] = "ndarray"
    libraries: ClassVar[List[ModuleType]] = [np]

    shape: Tuple[Optional[int], ...]
    dtype: str

    @staticmethod
    def _abstract_shape(shape):
        return (None,) + shape[1:]

    @classmethod
    def process(cls, obj, **kwargs) -> DatasetType:
        return NumpyNdarrayType(
            shape=cls._abstract_shape(obj.shape), dtype=obj.dtype.name
        )

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def deserialize(self, obj):
        try:
            ret = np.array(obj, dtype=np_type_from_string(self.dtype))
        except (ValueError, TypeError):
            raise DeserializationError(
                f"given object: {obj} could not be converted to array "
                f"of type: {np_type_from_string(self.dtype)}"
            )
        self._check_shape(ret, DeserializationError)
        return ret

    def _subtype(self, subshape: Tuple[Optional[int], ...]):
        if len(subshape) == 0:
            return python_type_from_np_string_repr(self.dtype)
        return conlist(self._subtype(subshape[1:]))

    def get_model(self) -> Type[BaseModel]:
        # TODO: https://github.com/iterative/mlem/issues/33
        return create_model(
            "NumpyNdarray", __root__=(List[self._subtype(self.shape[1:])], ...)  # type: ignore
        )

    def serialize(self, instance: np.ndarray):
        self._check_type(instance, np.ndarray, SerializationError)
        exp_type = np_type_from_string(self.dtype)
        if instance.dtype != exp_type:
            raise SerializationError(
                f"given array is of type: {instance.dtype}, expected: {exp_type}"
            )
        self._check_shape(instance, SerializationError)
        return instance.tolist()

    def _check_shape(self, array, exc_type):
        if tuple(array.shape)[1:] != self.shape[1:]:
            raise exc_type(
                f"given array is of shape: {(None,) + tuple(array.shape)[1:]}, expected: {self.shape}"
            )

    def get_writer(self):
        return NumpyArrayWriter()


DATA_FILE = "data.npz"
DATA_KEY = "data"


class NumpyArrayWriter(DatasetWriter):
    """DatasetWriter implementation for numpy ndarray"""

    type: ClassVar[str] = "numpy"

    def write(
        self, dataset: Dataset, fs: AbstractFileSystem, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        fs.makedirs(path, True)
        with fs.open(os.path.join(path, DATA_FILE), mode="wb") as f:
            np.savez_compressed(f, **{DATA_KEY: dataset.data})
        return NumpyArrayReader(dataset_type=dataset.dataset_type), [DATA_FILE]


class NumpyArrayReader(DatasetReader):
    """DatasetReader implementation for numpy ndarray"""

    type: ClassVar[str] = "numpy"

    def read(self, fs: AbstractFileSystem, path: str) -> Dataset:
        with fs.open(os.path.join(path, DATA_FILE)) as f:
            data = np.load(f)[DATA_KEY]
        return Dataset(data, self.dataset_type)
