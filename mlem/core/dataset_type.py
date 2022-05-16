"""
Base classes for working with datasets in MLEM
"""
import builtins
import posixpath
from abc import ABC, abstractmethod
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    List,
    Optional,
    Sized,
    Tuple,
    Type,
    Union,
)

import flatdict
from pydantic import BaseModel
from pydantic.main import create_model

from mlem.core.artifacts import Artifacts, Storage
from mlem.core.base import MlemABC
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.hooks import Analyzer, Hook
from mlem.core.requirements import Requirements, WithRequirements
from mlem.utils.module import get_object_requirements


class DatasetType(ABC, MlemABC, WithRequirements):
    """
    Base class for dataset type metadata.
    """

    class Config:
        type_root = True
        exclude = {"data"}

    abs_name: ClassVar[str] = "dataset_type"
    data: Any

    @staticmethod
    def check_type(obj, exp_type, exc_type):
        if not isinstance(obj, exp_type):
            raise exc_type(
                f"given dataset is of type: {type(obj)}, expected: {exp_type}"
            )

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

    def bind(self, data: Any):
        self.data = data
        return self

    @classmethod
    def create(cls, obj: Any, **kwargs):
        return DatasetAnalyzer.analyze(obj, **kwargs).bind(obj)


class DatasetSerializer(ABC):
    @abstractmethod
    def serialize(self, instance: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, obj: dict) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model(self, prefix: str = "") -> Union[Type[BaseModel], type]:
        raise NotImplementedError


class UnspecifiedDatasetType(DatasetType, DatasetSerializer):
    type: ClassVar = "unspecified"

    def serialize(self, instance: object) -> dict:
        return instance  # type: ignore

    def deserialize(self, obj: dict) -> object:
        return obj

    def get_requirements(self) -> Requirements:
        return Requirements()

    def get_writer(self, **kwargs) -> "DatasetWriter":
        raise NotImplementedError

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        raise NotImplementedError


class DatasetHook(Hook[DatasetType], ABC):
    pass


class DatasetAnalyzer(Analyzer):
    base_hook_class = DatasetHook


class DatasetReader(MlemABC, ABC):
    """"""

    class Config:
        type_root = True

    dataset_type: DatasetType
    abs_name: ClassVar[str] = "dataset_reader"

    @abstractmethod
    def read(self, artifacts: Artifacts) -> DatasetType:
        raise NotImplementedError

    @abstractmethod
    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DatasetType]:
        raise NotImplementedError


class DatasetWriter(MlemABC):
    """"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "dataset_writer"
    art_name: ClassVar[str] = "data"

    @abstractmethod
    def write(
        self, dataset: DatasetType, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
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
        return PrimitiveWriter(**kwargs)

    def get_requirements(self) -> Requirements:
        return Requirements.new()

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        return self.to_type


class PrimitiveWriter(DatasetWriter):
    type: ClassVar[str] = "primitive"

    def write(
        self, dataset: DatasetType, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        with storage.open(path) as (f, art):
            f.write(str(dataset.data).encode("utf-8"))
        return PrimitiveReader(dataset_type=dataset), {self.art_name: art}


class PrimitiveReader(DatasetReader):
    type: ClassVar[str] = "primitive"
    dataset_type: PrimitiveType

    def read(self, artifacts: Artifacts) -> DatasetType:
        if DatasetWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DatasetWriter.art_name} file"
            )
        with artifacts[DatasetWriter.art_name].open() as f:
            res = f.read().decode("utf-8")
            if res == "None":
                data = None
            elif res == "False":
                data = False
            else:
                data = self.dataset_type.to_type(res)
            return self.dataset_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DatasetType]:
        raise NotImplementedError


class ListDatasetType(DatasetType, DatasetSerializer):
    """
    DatasetType for list type
    for a list of elements with same types such as [1, 2, 3, 4, 5]
    """

    type: ClassVar[str] = "list"
    dtype: DatasetType
    size: Optional[int]

    def is_list(self):
        return True

    def list_size(self):
        return self.size

    def get_requirements(self) -> Requirements:
        return self.dtype.get_requirements()

    def deserialize(self, obj):
        _check_type_and_size(obj, list, self.size, DeserializationError)
        return [self.dtype.get_serializer().deserialize(o) for o in obj]

    def serialize(self, instance: list):
        _check_type_and_size(instance, list, self.size, SerializationError)
        return [self.dtype.get_serializer().serialize(o) for o in instance]

    def get_writer(self, **kwargs):
        return ListWriter(**kwargs)

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        subname = prefix + "__root__"
        return create_model(
            prefix + "ListDataset",
            __root__=(List[self.dtype.get_serializer().get_model(subname)], ...),  # type: ignore
        )


class ListWriter(DatasetWriter):
    type: ClassVar[str] = "list"

    def write(
        self, dataset: DatasetType, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        if not isinstance(dataset, ListDatasetType):
            raise ValueError(
                f"expected dataset to be of ListDatasetType, got {type(dataset)} instead"
            )
        res = {}
        readers = []
        for i, elem in enumerate(dataset.data):
            elem_reader, art = dataset.dtype.get_writer().write(
                dataset.dtype.copy().bind(elem),
                storage,
                posixpath.join(path, str(i)),
            )
            res[str(i)] = art
            readers.append(elem_reader)

        return ListReader(
            dataset_type=dataset, readers=readers
        ), flatdict.FlatterDict(res, delimiter="/")


class ListReader(DatasetReader):
    type: ClassVar[str] = "list"
    dataset_type: ListDatasetType
    readers: List[DatasetReader]

    def read(self, artifacts: Artifacts) -> DatasetType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_list = []
        for i, reader in enumerate(self.readers):
            elem_dtype = reader.read(artifacts[str(i)])  # type: ignore
            data_list.append(elem_dtype.data)
        return self.dataset_type.copy().bind(data_list)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DatasetType]:
        raise NotImplementedError


class _TupleLikeDatasetType(DatasetType, DatasetSerializer):
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
            t.get_serializer().deserialize(o) for t, o in zip(self.items, obj)
        )

    def serialize(self, instance: Sized):
        _check_type_and_size(
            instance, self.actual_type, len(self.items), SerializationError
        )
        return self.actual_type(
            t.get_serializer().serialize(o)
            for t, o in zip(self.items, instance)  # type: ignore
            # TODO: https://github.com/iterative/mlem/issues/33 inspect non-iterable sized
        )

    def get_requirements(self) -> Requirements:
        return sum(
            [i.get_requirements() for i in self.items], Requirements.new()
        )

    def get_writer(self, **kwargs) -> "DatasetWriter":
        return _TupleLikeDatasetWriter(**kwargs)

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        names = [f"{prefix}{i}_" for i in range(len(self.items))]
        return create_model(
            prefix + "_TupleLikeDataset",
            __root__=(
                Tuple[
                    tuple(
                        t.get_serializer().get_model(name)
                        for name, t in zip(names, self.items)
                    )
                ],
                ...,
            ),
        )


def _check_type_and_size(obj, dtype, size, exc_type):
    DatasetType.check_type(obj, dtype, exc_type)
    if size != -1 and len(obj) != size:
        raise exc_type(
            f"given {dtype.__name__} has len: {len(obj)}, expected: {size}"
        )


class _TupleLikeDatasetWriter(DatasetWriter):
    type: ClassVar[str] = "tuple_like"

    def write(
        self, dataset: DatasetType, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        if not isinstance(dataset, _TupleLikeDatasetType):
            raise ValueError(
                f"expected dataset to be of _TupleLikeDatasetType, got {type(dataset)} instead"
            )
        res = {}
        readers = []
        for i, (elem_dtype, elem) in enumerate(
            zip(dataset.items, dataset.data)
        ):
            elem_reader, art = elem_dtype.get_writer().write(
                elem_dtype.copy().bind(elem),
                storage,
                posixpath.join(path, str(i)),
            )
            res[str(i)] = art
            readers.append(elem_reader)

        return (
            _TupleLikeDatasetReader(dataset_type=dataset, readers=readers),
            flatdict.FlatterDict(res, delimiter="/"),
        )


class _TupleLikeDatasetReader(DatasetReader):
    type: ClassVar[str] = "tuple_like"
    dataset_type: _TupleLikeDatasetType
    readers: List[DatasetReader]

    def read(self, artifacts: Artifacts) -> DatasetType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_list = []
        for i, elem_reader in enumerate(self.readers):
            elem_dtype = elem_reader.read(artifacts[str(i)])  # type: ignore
            data_list.append(elem_dtype.data)
        data_list = self.dataset_type.actual_type(data_list)
        return self.dataset_type.copy().bind(data_list)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DatasetType]:
        raise NotImplementedError


class TupleLikeListDatasetType(_TupleLikeDatasetType):
    """
    DatasetType for tuple-like list type
    can be a list of elements with different types such as [1, False, 3.2, "mlem", None]
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


class DictDatasetType(DatasetType, DatasetSerializer, DatasetHook):
    """
    DatasetType for dict type
    """

    type: ClassVar[str] = "dict"
    item_types: Dict[str, DatasetType]

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, dict)

    @classmethod
    def process(cls, obj: Any, **kwargs) -> "DictDatasetType":
        return DictDatasetType(
            item_types={
                k: DatasetAnalyzer.analyze(v) for (k, v) in obj.items()
            }
        )

    def deserialize(self, obj):
        self._check_type_and_keys(obj, DeserializationError)
        return {
            k: self.item_types[k]
            .get_serializer()
            .deserialize(
                v,
            )
            for k, v in obj.items()
        }

    def serialize(self, instance: dict):
        self._check_type_and_keys(instance, SerializationError)
        return {
            k: self.item_types[k].get_serializer().serialize(v)
            for k, v in instance.items()
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

    def get_writer(self, **kwargs) -> "DatasetWriter":
        return DictWriter(**kwargs)

    def get_model(self, prefix="") -> Type[BaseModel]:
        kwargs = {
            k: (v.get_serializer().get_model(prefix + k + "_"), ...)
            for k, v in self.item_types.items()
        }
        return create_model(prefix + "DictDataset", **kwargs)  # type: ignore


class DictWriter(DatasetWriter):
    type: ClassVar[str] = "dict"

    def write(
        self, dataset: DatasetType, storage: Storage, path: str
    ) -> Tuple[DatasetReader, Artifacts]:
        if not isinstance(dataset, DictDatasetType):
            raise ValueError(
                f"expected dataset to be of DictDatasetType, got {type(dataset)} instead"
            )
        res = {}
        readers = {}
        for (key, dtype) in dataset.item_types.items():
            dtype_reader, art = dtype.get_writer().write(
                dtype.copy().bind(dataset.data[key]),
                storage,
                posixpath.join(path, key),
            )
            res[key] = art
            readers[key] = dtype_reader
        return DictReader(
            dataset_type=dataset, item_readers=readers
        ), flatdict.FlatterDict(res, delimiter="/")


class DictReader(DatasetReader):
    type: ClassVar[str] = "dict"
    dataset_type: DictDatasetType
    item_readers: Dict[str, DatasetReader]

    def read(self, artifacts: Artifacts) -> DatasetType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_dict = {}
        for (key, dtype_reader) in self.item_readers.items():
            v_dataset_type = dtype_reader.read(artifacts[key])  # type: ignore
            data_dict[key] = v_dataset_type.data
        return self.dataset_type.copy().bind(data_dict)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DatasetType]:
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
