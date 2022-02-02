"""
Base classes for working with datasets in MLEM
"""
import builtins
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sized, Tuple, Type

from pydantic import BaseModel
from pydantic.main import create_model

from mlem.core.artifacts import Artifacts, Storage
from mlem.core.base import MlemObject
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import Analyzer, Hook
from mlem.core.requirements import Requirements, WithRequirements
from mlem.utils.module import get_object_requirements


class DatasetType(ABC, MlemObject, WithRequirements):
    """
    Base class for dataset type metadata.
    """

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "dataset_type"

    @staticmethod
    def check_type(obj, exp_type, exc_type):
        if not isinstance(obj, exp_type):
            raise exc_type(
                f"given dataset is of type: {type(obj)}, expected: {exp_type}"
            )

    @abstractmethod
    def serialize(self, instance: Any) -> dict:
        """"""

    @abstractmethod
    def deserialize(self, obj: dict) -> Any:
        """"""

    @abstractmethod
    def get_requirements(self) -> Requirements:
        """"""  # TODO: https://github.com/iterative/mlem/issues/16 docs
        return get_object_requirements(self)

    @abstractmethod
    def get_writer(self, **kwargs) -> "DatasetWriter":
        raise NotImplementedError

    def get_serializer(
        self, **kwargs  # pylint: disable=unused-argument
    ) -> "DatasetSerializer":
        if isinstance(self, DatasetSerializer):
            return self
        raise NotImplementedError


class UnspecifiedDatasetType(DatasetType):
    def serialize(self, instance: object) -> dict:
        return instance  # type: ignore

    def deserialize(self, obj: dict) -> object:
        return obj

    def get_requirements(self) -> Requirements:
        return Requirements()

    def get_writer(self, **kwargs) -> "DatasetWriter":
        raise NotImplementedError


class DatasetHook(Hook[DatasetType], ABC):
    pass


class DatasetAnalyzer(Analyzer):
    base_hook_class = DatasetHook


class DatasetSerializer(ABC):
    @abstractmethod
    def serialize(self, obj: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, payload: dict) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model(self) -> Type[BaseModel]:
        raise NotImplementedError


class PrimitiveType(DatasetType, DatasetHook, DatasetSerializer):
    """
    DatasetType for int, str, bool, complex and float types
    """

    PRIMITIVES: ClassVar[set] = {int, str, bool, complex, float, type(None)}
    type: ClassVar[str] = "primitive"

    ptype: str

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return type(obj) in cls.PRIMITIVES

    @classmethod
    def process(cls, obj: Any, **kwargs) -> "PrimitiveType":
        return PrimitiveType(ptype=type(obj).__name__)

    @property
    def to_type(self):
        if self.ptype == "NoneType":
            return type(None)
        return getattr(builtins, self.ptype)

    def deserialize(self, obj):
        return self.to_type(obj)

    def serialize(self, instance):
        self.check_type(instance, self.to_type, ValueError)
        return instance

    def get_writer(self, **kwargs):
        raise NotImplementedError  # TODO: https://github.com/iterative/mlem/issues/35

    def get_requirements(self) -> Requirements:
        return Requirements.new()

    def get_model(self) -> Type[BaseModel]:
        return create_model("Primitive", __root__=self.to_type)


class ListTypeWithSpec(DatasetType):
    """
    Abstract base class for `list`-like types providing its OpenAPI schema definition
    """

    def is_list(self):
        return True

    @abstractmethod
    def list_size(self):
        raise NotImplementedError  # pragma: no cover


class SizedTypedListType(ListTypeWithSpec):
    """
    Subclass of :class:`ListTypeWithSpec` which specifies size of internal `list`
    """

    dtype: DatasetType
    size: Optional[int]

    def list_size(self):
        return self.size


class ListDatasetType(SizedTypedListType):
    """
    DatasetType for list type
    """

    type: ClassVar[str] = "list"

    def get_requirements(self) -> Requirements:
        return self.dtype.get_requirements()

    def deserialize(self, obj):
        _check_type_and_size(obj, list, self.size, DeserializationError)
        return [self.dtype.deserialize(o) for o in obj]

    def serialize(self, instance: list):
        _check_type_and_size(instance, list, self.size, SerializationError)
        return [self.dtype.serialize(o) for o in instance]

    def get_writer(self, **kwargs):
        raise NotImplementedError


class _TupleLikeDatasetType(DatasetType):
    """
    DatasetType for tuple-like collections
    """

    items: List[DatasetType]
    actual_type: ClassVar[type]

    def deserialize(self, obj):
        _check_type_and_size(
            obj, self.actual_type, len(self.items), DeserializationError
        )
        return self.actual_type(
            t.deserialize(o) for t, o in zip(self.items, obj)
        )

    def serialize(self, instance: Sized):
        _check_type_and_size(
            instance, self.actual_type, len(self.items), SerializationError
        )
        return self.actual_type(
            t.serialize(o)
            for t, o in zip(self.items, instance)  # type: ignore
            # TODO: https://github.com/iterative/mlem/issues/33 inspect non-iterable sized
        )

    def get_requirements(self) -> Requirements:
        return sum(
            [i.get_requirements() for i in self.items], Requirements.new()
        )

    def get_writer(self, **kwargs):
        raise NotImplementedError


def _check_type_and_size(obj, dtype, size, exc_type):
    DatasetType.check_type(obj, dtype, exc_type)
    if size != -1 and len(obj) != size:
        raise exc_type(
            f"given {dtype.__name__} has len: {len(obj)}, expected: {size}"
        )


class TupleLikeListDatasetType(_TupleLikeDatasetType):
    """
    DatasetType for tuple-like list type
    """

    actual_type: ClassVar = list
    type: ClassVar[str] = "tuple_like_list"


class TupleDatasetType(_TupleLikeDatasetType):
    """
    DatasetType for tuple type
    """

    actual_type: ClassVar = tuple
    type: ClassVar[str] = "tuple"


class OrderedCollectionHookDelegator(DatasetHook):
    """
    Hook for list/tuple data
    """

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, (list, tuple))

    @classmethod
    def process(cls, obj, **kwargs) -> DatasetType:
        if isinstance(obj, tuple):
            return TupleDatasetType(
                items=[DatasetAnalyzer.analyze(o) for o in obj]
            )

        py_types = {type(o) for o in obj}
        if len(obj) <= 1 or len(py_types) > 1:
            return TupleLikeListDatasetType(
                items=[DatasetAnalyzer.analyze(o) for o in obj]
            )

        if not py_types.intersection(
            PrimitiveType.PRIMITIVES
        ):  # py_types is guaranteed to be singleton set here
            return TupleLikeListDatasetType(
                items=[DatasetAnalyzer.analyze(o) for o in obj]
            )

        # optimization for large lists of same primitive type elements
        return ListDatasetType(
            dtype=DatasetAnalyzer.analyze(obj[0]), size=len(obj)
        )


class DictDatasetType(DatasetType):
    """
    DatasetType for dict type
    """

    type: ClassVar[str] = "dict"
    item_types: Dict[str, DatasetType]

    def deserialize(self, obj):
        self._check_type_and_keys(obj, DeserializationError)
        return {
            k: self.item_types[k].deserialize(
                v,
            )
            for k, v in obj.items()
        }

    def serialize(self, instance: dict):
        self._check_type_and_keys(instance, SerializationError)
        return {
            k: self.item_types[k].serialize(v) for k, v in instance.items()
        }

    def _check_type_and_keys(self, obj, exc_type):
        self.check_type(obj, dict, exc_type)
        if set(obj.keys()) != set(self.item_types.keys()):
            raise exc_type(
                f"given dict has keys: {set(obj.keys())}, expected: {set(self.item_types.keys())}"
            )

    def get_requirements(self) -> Requirements:
        return sum(
            [i.get_requirements() for i in self.item_types.values()],
            Requirements.new(),
        )

    def get_writer(self, **kwargs):
        raise NotImplementedError


#
#
# class BytesDatasetType(DatasetType):
#     """
#     DatasetType for bytes objects
#     """
#     type = 'bytes'
#     real_type = None
#
#     def __init__(self):
#         pass
#
#     def get_spec(self) -> ArgList:
#         return [Field('file', bytes, False)]
#
#     def deserialize(self, obj) -> object:
#         return obj
#
#     def serialize(self, instance: object) -> dict:
#         return instance
#
#     @property
#     def requirements(self) -> Requirements:
#         return Requirements()
#
#     def get_writer(self):
#         return PickleWriter()


@dataclass
class Dataset:
    data: Any
    dataset_type: DatasetType

    @classmethod
    def create(cls, data: Any):
        return Dataset(data, DatasetAnalyzer.analyze(data))


class DatasetReader(MlemObject, ABC):
    class Config:
        type_root = True

    dataset_type: DatasetType
    abs_name: ClassVar[str] = "dataset_reader"

    @abstractmethod
    def read(self, artifacts: Artifacts) -> Dataset:
        raise NotImplementedError


class DatasetWriter(MlemObject):
    class Config:
        type_root = True

    abs_name: ClassVar[str] = "dataset_writer"

    @abstractmethod
    def write(
        self, dataset: Dataset, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        raise NotImplementedError
