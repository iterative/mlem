"""
Base classes for working with data in MLEM
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


class DataType(ABC, MlemABC, WithRequirements):
    """
    Base class for data metadata
    """

    class Config:
        type_root = True
        exclude = {"data"}

    abs_name: ClassVar[str] = "data_type"
    data: Any

    @staticmethod
    def check_type(obj, exp_type, exc_type):
        if not isinstance(obj, exp_type):
            raise exc_type(
                f"given data is of type: {type(obj)}, expected: {exp_type}"
            )

    @abstractmethod
    def get_requirements(self) -> Requirements:
        return get_object_requirements(self)

    @abstractmethod
    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        raise NotImplementedError

    def get_serializer(
        self, **kwargs  # pylint: disable=unused-argument
    ) -> "DataSerializer":
        if isinstance(self, DataSerializer):
            return self
        raise NotImplementedError

    def bind(self, data: Any):
        self.data = data
        return self

    @classmethod
    def create(cls, obj: Any, **kwargs):
        return DataAnalyzer.analyze(obj, **kwargs).bind(obj)


class DataSerializer(ABC):
    """Base class for data-to-dict serialization logic"""

    @abstractmethod
    def serialize(self, instance: Any) -> dict:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, obj: dict) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model(self, prefix: str = "") -> Union[Type[BaseModel], type]:
        raise NotImplementedError


class UnspecifiedDataType(DataType, DataSerializer):
    """Special data type for cases when it's not provided"""

    type: ClassVar = "unspecified"

    def serialize(self, instance: object) -> dict:
        return instance  # type: ignore

    def deserialize(self, obj: dict) -> object:
        return obj

    def get_requirements(self) -> Requirements:
        return Requirements()

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        raise NotImplementedError

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        raise NotImplementedError


class DataHook(Hook[DataType], ABC):
    """Base class for hooks to analyze data objects"""


class DataAnalyzer(Analyzer):
    """Analyzer for data objects"""

    base_hook_class = DataHook


class DataReader(MlemABC, ABC):
    """Base class for defining logic to read data from a set of `Artifact`s"""

    class Config:
        type_root = True

    data_type: DataType
    abs_name: ClassVar[str] = "data_reader"

    @abstractmethod
    def read(self, artifacts: Artifacts) -> DataType:
        raise NotImplementedError

    @abstractmethod
    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class DataWriter(MlemABC):
    """Base class for defining logic to write data. Should produce a set of
    `Artifact`s and a corresponding reader"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "data_writer"
    art_name: ClassVar[str] = "data"

    @abstractmethod
    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        raise NotImplementedError


class PrimitiveType(DataType, DataHook, DataSerializer):
    """
    DataType for int, str, bool, complex and float types
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

    def get_writer(self, project: str = None, filename: str = None, **kwargs):
        return PrimitiveWriter(**kwargs)

    def get_requirements(self) -> Requirements:
        return Requirements.new()

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        return self.to_type


class PrimitiveWriter(DataWriter):
    type: ClassVar[str] = "primitive"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            f.write(str(data.data).encode("utf-8"))
        return PrimitiveReader(data_type=data), {self.art_name: art}


class PrimitiveReader(DataReader):
    type: ClassVar[str] = "primitive"
    data_type: PrimitiveType

    def read(self, artifacts: Artifacts) -> DataType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            res = f.read().decode("utf-8")
            if res == "None":
                data = None
            elif res == "False":
                data = False
            else:
                data = self.data_type.to_type(res)
            return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class ArrayType(DataType, DataSerializer):
    """
    DataType for lists with elements of the same type such as [1, 2, 3, 4, 5]
    """

    type: ClassVar[str] = "array"
    dtype: DataType
    size: Optional[int]

    def get_requirements(self) -> Requirements:
        return self.dtype.get_requirements()

    def deserialize(self, obj):
        _check_type_and_size(obj, list, self.size, DeserializationError)
        return [self.dtype.get_serializer().deserialize(o) for o in obj]

    def serialize(self, instance: list):
        _check_type_and_size(instance, list, self.size, SerializationError)
        return [self.dtype.get_serializer().serialize(o) for o in instance]

    def get_writer(self, project: str = None, filename: str = None, **kwargs):
        return ArrayWriter(**kwargs)

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        subname = prefix + "__root__"
        return create_model(
            prefix + "Array",
            __root__=(List[self.dtype.get_serializer().get_model(subname)], ...),  # type: ignore
        )


class ArrayWriter(DataWriter):
    type: ClassVar[str] = "array"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        if not isinstance(data, ArrayType):
            raise ValueError(
                f"expected data to be of ArrayType, got {type(data)} instead"
            )
        res = {}
        readers = []
        for i, elem in enumerate(data.data):
            elem_reader, art = data.dtype.get_writer().write(
                data.dtype.copy().bind(elem),
                storage,
                posixpath.join(path, str(i)),
            )
            res[str(i)] = art
            readers.append(elem_reader)

        return ArrayReader(data_type=data, readers=readers), dict(
            flatdict.FlatterDict(res, delimiter="/")
        )


class ArrayReader(DataReader):
    type: ClassVar[str] = "array"
    data_type: ArrayType
    readers: List[DataReader]

    def read(self, artifacts: Artifacts) -> DataType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_list = []
        for i, reader in enumerate(self.readers):
            elem_dtype = reader.read(artifacts[str(i)])  # type: ignore
            data_list.append(elem_dtype.data)
        return self.data_type.copy().bind(data_list)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class _TupleLikeType(DataType, DataSerializer):
    """
    DataType for tuple-like collections
    """

    items: List[DataType]
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

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        return _TupleLikeWriter(**kwargs)

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        names = [f"{prefix}{i}_" for i in range(len(self.items))]
        return create_model(
            prefix + "_TupleLikeType",
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
    DataType.check_type(obj, dtype, exc_type)
    if size != -1 and len(obj) != size:
        raise exc_type(
            f"given {dtype.__name__} has len: {len(obj)}, expected: {size}"
        )


class _TupleLikeWriter(DataWriter):
    type: ClassVar[str] = "tuple_like"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        if not isinstance(data, _TupleLikeType):
            raise ValueError(
                f"expected data to be of _TupleLikeDataType, got {type(data)} instead"
            )
        res = {}
        readers = []
        for i, (elem_dtype, elem) in enumerate(zip(data.items, data.data)):
            elem_reader, art = elem_dtype.get_writer().write(
                elem_dtype.copy().bind(elem),
                storage,
                posixpath.join(path, str(i)),
            )
            res[str(i)] = art
            readers.append(elem_reader)

        return (
            _TupleLikeReader(data_type=data, readers=readers),
            dict(flatdict.FlatterDict(res, delimiter="/")),
        )


class _TupleLikeReader(DataReader):
    type: ClassVar[str] = "tuple_like"
    data_type: _TupleLikeType
    readers: List[DataReader]

    def read(self, artifacts: Artifacts) -> DataType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_list = []
        for i, elem_reader in enumerate(self.readers):
            elem_dtype = elem_reader.read(artifacts[str(i)])  # type: ignore
            data_list.append(elem_dtype.data)
        data_list = self.data_type.actual_type(data_list)
        return self.data_type.copy().bind(data_list)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class ListType(_TupleLikeType):
    """
    DataType for list with separate type for each element
    such as [1, False, 3.2, "mlem", None]
    """

    actual_type: ClassVar = list
    type: ClassVar[str] = "list"


class TupleType(_TupleLikeType):
    """
    DataType for tuple type
    """

    actual_type: ClassVar = tuple
    type: ClassVar[str] = "tuple"


class OrderedCollectionHook(DataHook):
    """
    Hook for list/tuple data
    """

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, (list, tuple))

    @classmethod
    def process(cls, obj, **kwargs) -> DataType:
        if isinstance(obj, tuple):
            return TupleType(items=[DataAnalyzer.analyze(o) for o in obj])

        py_types = {type(o) for o in obj}
        if len(obj) <= 1 or len(py_types) > 1:
            return ListType(items=[DataAnalyzer.analyze(o) for o in obj])

        if not py_types.intersection(
            PrimitiveType.PRIMITIVES
        ):  # py_types is guaranteed to be singleton set here
            items_types = [DataAnalyzer.analyze(o) for o in obj]
            first, *others = items_types
            for other in others:
                if first != other:
                    return ListType(items=items_types)
            return ArrayType(dtype=first, size=len(obj))

        # optimization for large lists of same primitive type elements
        return ArrayType(dtype=DataAnalyzer.analyze(obj[0]), size=len(obj))


class DictType(DataType, DataSerializer, DataHook):
    """
    DataType for dict
    """

    type: ClassVar[str] = "dict"
    item_types: Dict[str, DataType]

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, dict)

    @classmethod
    def process(cls, obj: Any, **kwargs) -> "DictType":
        return DictType(
            item_types={k: DataAnalyzer.analyze(v) for (k, v) in obj.items()}
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

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        return DictWriter(**kwargs)

    def get_model(self, prefix="") -> Type[BaseModel]:
        kwargs = {
            k: (v.get_serializer().get_model(prefix + k + "_"), ...)
            for k, v in self.item_types.items()
        }
        return create_model(prefix + "DictType", **kwargs)  # type: ignore


class DictWriter(DataWriter):
    type: ClassVar[str] = "dict"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        if not isinstance(data, DictType):
            raise ValueError(
                f"expected data to be of DictType, got {type(data)} instead"
            )
        res = {}
        readers = {}
        for (key, dtype) in data.item_types.items():
            dtype_reader, art = dtype.get_writer().write(
                dtype.copy().bind(data.data[key]),
                storage,
                posixpath.join(path, key),
            )
            res[key] = art
            readers[key] = dtype_reader
        return DictReader(data_type=data, item_readers=readers), dict(
            flatdict.FlatterDict(res, delimiter="/")
        )


class DictReader(DataReader):
    type: ClassVar[str] = "dict"
    data_type: DictType
    item_readers: Dict[str, DataReader]

    def read(self, artifacts: Artifacts) -> DataType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_dict = {}
        for (key, dtype_reader) in self.item_readers.items():
            v_data_type = dtype_reader.read(artifacts[key])  # type: ignore
            data_dict[key] = v_data_type.data
        return self.data_type.copy().bind(data_dict)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


#
#
# class BytesDataType(DataType):
#     """
#     DataType for bytes objects
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
