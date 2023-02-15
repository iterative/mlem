"""
Base classes for working with data in MLEM
"""
import builtins
import contextlib
import io
import json
import posixpath
from abc import ABC, abstractmethod
from typing import (
    Any,
    BinaryIO,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sized,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import flatdict
from pydantic import BaseModel, StrictInt, StrictStr, validator
from pydantic.main import create_model

from mlem.core.artifacts import (
    Artifact,
    Artifacts,
    InMemoryArtifact,
    InMemoryFileobjArtifact,
    InMemoryStorage,
    Storage,
)
from mlem.core.base import MlemABC
from mlem.core.errors import (
    DeserializationError,
    MlemError,
    SerializationError,
)
from mlem.core.hooks import Analyzer, Hook, IsInstanceHookMixin
from mlem.core.requirements import Requirements, WithRequirements
from mlem.utils.module import get_object_requirements

DDT = TypeVar("DDT")

T = TypeVar("T", bound="DataType")


class DataType(ABC, MlemABC, WithRequirements, Generic[DDT]):
    """
    Base class for data metadata
    """

    class Config:
        type_root = True
        exclude = {"value"}

    type: ClassVar[str]
    abs_name: ClassVar[str] = "data_type"
    value: Optional[DDT]

    @property
    def data(self) -> DDT:
        if self.value is None:
            raise MlemError("Cannot access data on empty data type object")
        return self.value

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
    ) -> "DataTypeSerializer":
        raise NotImplementedError

    def bind(self: T, data: Any) -> T:
        self.value = data
        return self

    @classmethod
    def create(cls, obj: Any, is_dynamic: bool = False, **kwargs):
        return DataAnalyzer.analyze(obj, is_dynamic=is_dynamic, **kwargs).bind(
            obj
        )

    def serialize(self, instance):
        return self.get_serializer().serialize(instance)

    def deserialize(self, obj):
        return self.get_serializer().deserialize(obj)

    def get_model(self, prefix: str = "", **kwargs):
        return self.get_serializer(**kwargs).get_model(prefix=prefix)


DT = TypeVar("DT", bound=DataType)
JsonTypes = Union[dict, list, int, str, bool, float, None]


class Serializer(MlemABC, Generic[DT]):
    """Base class for data-to-dict serialization logic"""

    class Config:
        type_root = True

    abs_name: ClassVar = "serializer"
    data_class: ClassVar[Type[DataType]]
    is_default: ClassVar[bool] = False

    @abstractmethod
    def serialize(self, data_type: DT, instance: Any):
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data_type: DT, obj) -> Any:
        raise NotImplementedError

    @property
    def binary(self) -> "BinarySerializer":
        if not isinstance(self, BinarySerializer):
            raise NotImplementedError
        return self

    @property
    def data(self) -> "DataSerializer":
        if not isinstance(self, DataSerializer):
            raise NotImplementedError
        return self

    @property
    def is_binary(self):
        return isinstance(self, BinarySerializer)

    def __init_subclass__(cls):
        if cls.is_default and issubclass(
            cls.data_class, WithDefaultSerializer
        ):
            cls.data_class.serializer_class = cls
            cls.type = cls.data_class.type
        super().__init_subclass__()


class DataTypeSerializer(BaseModel):
    serializer: Serializer
    """Serializer object"""
    data_type: DataType
    """DataType instance to use for serialization"""

    def serialize(self, instance: Any):
        return self.serializer.serialize(self.data_type, instance)

    def deserialize(self, obj) -> Any:
        return self.serializer.deserialize(self.data_type, obj)

    @contextlib.contextmanager
    def dump(self, instance: Any) -> Iterator[BinaryIO]:
        if not self.serializer.is_binary:
            raise NotImplementedError
        with self.serializer.binary.dump(self.data_type, instance) as f:
            yield f

    def get_model(self, prefix: str = "") -> Union[Type[BaseModel], type]:
        if self.serializer.is_binary:
            raise NotImplementedError
        return self.serializer.data.get_model(self.data_type, prefix)

    @property
    def support_files(self):
        return self.serializer.binary.support_files


class DefaultDataTypeSerializer(DataTypeSerializer):
    @validator("data_type")
    def is_correct_data_type(  # pylint: disable=no-self-argument
        cls, value, values  # noqa: B902
    ):
        serializer = values["serializer"]
        if isinstance(value, WithDefaultSerializer) and not isinstance(
            serializer, value.serializer_class
        ):
            raise ValueError(f"Cannot use {serializer} with {value.__class__}")
        return value


class DataSerializer(Serializer[DT], Generic[DT], ABC):
    """Base class for serializers to JSON data"""

    @abstractmethod
    def serialize(self, data_type: DT, instance: Any) -> JsonTypes:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data_type: DT, obj: JsonTypes) -> Any:
        raise NotImplementedError

    @abstractmethod
    def get_model(
        self, data_type: DT, prefix: str = ""
    ) -> Union[Type[BaseModel], type]:
        raise NotImplementedError


class BinarySerializer(Serializer[DT], Generic[DT], ABC):
    """Base class for serializers from/to raw binary data"""

    support_files: ClassVar[bool] = False

    @abstractmethod
    def serialize(self, data_type: DT, instance: Any) -> bytes:
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def dump(self, data_type: DT, instance: Any) -> Iterator[BinaryIO]:
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data_type: DT, obj: Union[bytes, BinaryIO]) -> Any:
        raise NotImplementedError


class UnspecifiedDataType(DataType):
    """Special data type for cases when it's not provided"""

    type: ClassVar = "unspecified"

    def get_serializer(self, **kwargs) -> "DataTypeSerializer":
        raise NotImplementedError

    def get_requirements(self) -> Requirements:
        return Requirements()

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        raise NotImplementedError


class DataHook(Hook[DataType], ABC):
    """Base class for hooks to analyze data objects"""


class DataAnalyzer(Analyzer):
    """Analyzer for data objects"""

    base_hook_class = DataHook


class DataReader(MlemABC, ABC, Generic[DT]):
    """Base class for defining logic to read data from a set of `Artifact`s"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "data_reader"
    data_type: DT
    """Resulting data type"""

    @abstractmethod
    def read(self, artifacts: Artifacts) -> DT:
        raise NotImplementedError

    @abstractmethod
    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DT]:
        raise NotImplementedError


class DataWriter(MlemABC, Generic[DT]):
    """Base class for defining logic to write data. Should produce a set of
    `Artifact`s and a corresponding reader"""

    class Config:
        type_root = True

    abs_name: ClassVar[str] = "data_writer"
    art_name: ClassVar[str] = "data"
    reader_class: ClassVar[Type[DataReader]]

    @abstractmethod
    def write(
        self, data: DT, storage: Storage, path: str
    ) -> Tuple[DataReader[DT], Artifacts]:
        raise NotImplementedError

    def get_reader(self, data_type: DT) -> DataReader[DT]:
        if not hasattr(self, "reader_class"):
            raise NotImplementedError
        try:
            return self.reader_class(data_type=data_type)
        except ValueError as e:
            raise NotImplementedError from e


class WithDefaultSerializer:
    """Mixin to use for DataTypes with default serializer implementations"""

    serializer_class: ClassVar[Type[Serializer]]

    def get_serializer(
        self, **kwargs  # pylint: disable=unused-argument
    ) -> "DataTypeSerializer":
        serializer = self.serializer_class()
        if not isinstance(self, DataType):
            raise ValueError(
                "WithDefaultSerializer should be used as a mixin with DataType"
            )
        return DefaultDataTypeSerializer(serializer=serializer, data_type=self)


class PrimitiveType(WithDefaultSerializer, DataType, DataHook):
    """
    DataType for int, str, bool, complex and float types
    """

    PRIMITIVES: ClassVar[set] = {int, str, bool, complex, float, type(None)}
    type: ClassVar[str] = "primitive"

    ptype: str
    """Name of builtin type"""

    @property
    def data(self) -> Any:
        if self.value is None and self.ptype != "NoneType":
            raise MlemError("Cannot access data on empty data type object")
        return self.value

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return type(obj) in cls.PRIMITIVES

    @classmethod
    def process(cls, obj: Any, **kwargs) -> "PrimitiveType":
        return PrimitiveType(ptype=type(obj).__name__)

    def get_writer(self, project: str = None, filename: str = None, **kwargs):
        return PrimitiveWriter(**kwargs)

    def get_requirements(self) -> Requirements:
        return Requirements.new()

    @property
    def to_type(self):
        if self.ptype == "NoneType":
            return type(None)
        return getattr(builtins, self.ptype)


class PrimitiveSerializer(DataSerializer[PrimitiveType]):
    """Serialzier for primitive types"""

    data_class: ClassVar = PrimitiveType
    is_default: ClassVar = True

    def get_model(
        self, data_type, prefix: str = ""
    ) -> Union[Type[BaseModel], type]:
        return data_type.to_type

    def deserialize(self, data_type, obj):
        return data_type.to_type(obj)

    def serialize(self, data_type, instance):
        data_type.check_type(instance, data_type.to_type, ValueError)
        return instance


class PrimitiveReader(DataReader[PrimitiveType]):
    """Reader for primitive types"""

    type: ClassVar[str] = "primitive"
    data_type: PrimitiveType

    def read(self, artifacts: Artifacts) -> PrimitiveType:
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
    ) -> Iterator[PrimitiveType]:
        raise NotImplementedError


class PrimitiveWriter(DataWriter[PrimitiveType]):
    """Writer for primitive types"""

    type: ClassVar[str] = "primitive"
    reader_class: ClassVar = PrimitiveReader

    def write(
        self, data: PrimitiveType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            f.write(str(data.data).encode("utf-8"))
        return PrimitiveReader(data_type=data), {self.art_name: art}


class ArrayType(WithDefaultSerializer, DataType):
    """
    DataType for lists with elements of the same type such as [1, 2, 3, 4, 5]
    """

    type: ClassVar[str] = "array"
    dtype: DataType
    """DataType of elements"""
    size: Optional[int]
    """Size of the list"""

    def get_requirements(self) -> Requirements:
        return self.dtype.get_requirements()

    def get_writer(self, project: str = None, filename: str = None, **kwargs):
        return ArrayWriter(**kwargs)


class ArraySerializer(DataSerializer[ArrayType]):
    """Serialzier for list with single element type"""

    data_class: ClassVar = ArrayType
    is_default: ClassVar = True

    def get_model(self, data_type, prefix: str = "") -> Type[BaseModel]:
        subname = prefix + "__root__"
        model = data_type.dtype.get_serializer().get_model(subname)
        item_type = List[model]  # type: ignore[valid-type]
        return create_model(
            prefix + "Array",
            __root__=(item_type, ...),
        )

    def deserialize(self, data_type, obj):
        _check_type_and_size(obj, list, data_type.size, DeserializationError)
        return [data_type.dtype.get_serializer().deserialize(o) for o in obj]

    def serialize(self, data_type, instance: list):
        _check_type_and_size(
            instance, list, data_type.size, SerializationError
        )
        return [
            data_type.dtype.get_serializer().serialize(o) for o in instance
        ]


class ArrayWriter(DataWriter):
    """Writer for lists with single element type"""

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
    """Reader for lists with single element type"""

    type: ClassVar[str] = "array"
    data_type: ArrayType
    readers: List[DataReader]
    """Inner readers"""

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


class _TupleLikeType(WithDefaultSerializer, DataType):
    """
    DataType for tuple-like collections
    """

    type: ClassVar = "_tuple_like"
    actual_type: ClassVar[type]

    items: List[DataType]
    """DataTypes of elements"""

    def get_requirements(self) -> Requirements:
        return sum(
            [i.get_requirements() for i in self.items], Requirements.new()
        )

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        return _TupleLikeWriter(**kwargs)


class _TupleLikeSerializer(DataSerializer[_TupleLikeType]):
    """Serialzier for tuple-like sequence"""

    data_class: ClassVar = _TupleLikeType
    is_default: ClassVar = True

    def deserialize(self, data_type, obj):
        _check_type_and_size(
            obj,
            data_type.actual_type,
            len(data_type.items),
            DeserializationError,
        )
        return data_type.actual_type(
            t.get_serializer().deserialize(o)
            for t, o in zip(data_type.items, obj)
        )

    def serialize(self, data_type, instance: Sized):
        _check_type_and_size(
            instance,
            data_type.actual_type,
            len(data_type.items),
            SerializationError,
        )
        return data_type.actual_type(
            t.get_serializer().serialize(o)
            for t, o in zip(data_type.items, instance)  # type: ignore[call-overload]
            # TODO: https://github.com/iterative/mlem/issues/33 inspect non-iterable sized
        )

    def get_model(self, data_type, prefix: str = "") -> Type[BaseModel]:
        names = [f"{prefix}{i}_" for i in range(len(data_type.items))]
        return create_model(
            prefix + "_TupleLikeType",
            __root__=(
                Tuple[
                    tuple(
                        t.get_model(name)
                        for name, t in zip(names, data_type.items)
                    )
                ],
                ...,
            ),
        )


def _check_type_and_size(obj, dtype, size, exc_type):
    DataType.check_type(obj, dtype, exc_type)
    if size is not None and len(obj) != size:
        raise exc_type(
            f"given {dtype.__name__} has len: {len(obj)}, expected: {size}"
        )


class _TupleLikeWriter(DataWriter):
    """Writer for tuple-like data"""

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
    """Reader for tuple-like data"""

    type: ClassVar[str] = "tuple_like"
    data_type: _TupleLikeType
    readers: List[DataReader]
    """Inner readers"""

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
    def process(cls, obj, is_dynamic: bool = False, **kwargs) -> DataType:
        if isinstance(obj, tuple):
            return TupleType(
                items=[
                    DataAnalyzer.analyze(o, is_dynamic=is_dynamic, **kwargs)
                    for o in obj
                ]
            )

        py_types = {type(o) for o in obj}
        if len(obj) <= 1 or len(py_types) > 1:
            return ListType(
                items=[
                    DataAnalyzer.analyze(o, is_dynamic=is_dynamic, **kwargs)
                    for o in obj
                ]
            )

        size = None if is_dynamic else len(obj)

        if not py_types.intersection(
            PrimitiveType.PRIMITIVES
        ):  # py_types is guaranteed to be singleton set here
            items_types = [
                DataAnalyzer.analyze(o, is_dynamic=is_dynamic, **kwargs)
                for o in obj
            ]
            first, *others = items_types
            for other in others:
                if first != other:
                    return ListType(items=items_types)
            return ArrayType(dtype=first, size=size)

        # optimization for large lists of same primitive type elements
        return ArrayType(
            dtype=DataAnalyzer.analyze(
                obj[0], is_dynamic=is_dynamic, **kwargs
            ),
            size=size,
        )


class DictTypeHook(DataHook):
    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, dict)

    @classmethod
    def process(
        cls, obj: Any, is_dynamic: bool = False, **kwargs
    ) -> Union["DictType", "DynamicDictType"]:
        if not is_dynamic:
            return DictType.process(obj, **kwargs)
        return DynamicDictType.process(obj, **kwargs)


class DictType(WithDefaultSerializer, DataType):
    """
    DataType for dict with fixed set of keys
    """

    type: ClassVar[str] = "dict"
    item_types: Dict[Union[StrictStr, StrictInt], DataType]
    """Mapping key -> nested data type"""

    @classmethod
    def process(cls, obj, **kwargs):
        return DictType(
            item_types={
                k: DataAnalyzer.analyze(v, is_dynamic=False, **kwargs)
                for (k, v) in obj.items()
            }
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


class DictSerializer(DataSerializer[DictType]):
    """Serialzier for dict"""

    data_class: ClassVar = DictType
    is_default: ClassVar = True

    @staticmethod
    def _check_type_and_keys(data_type, obj, exc_type):
        data_type.check_type(obj, dict, exc_type)
        if set(obj.keys()) != set(data_type.item_types.keys()):
            raise exc_type(
                f"given dict has keys: {set(obj.keys())}, expected: {set(data_type.item_types.keys())}"
            )

    def deserialize(self, data_type, obj):
        self._check_type_and_keys(data_type, obj, DeserializationError)
        return {
            k: data_type.item_types[k]
            .get_serializer()
            .deserialize(
                v,
            )
            for k, v in obj.items()
        }

    def serialize(self, data_type, instance: dict):
        self._check_type_and_keys(data_type, instance, SerializationError)
        return {
            k: data_type.item_types[k].get_serializer().serialize(v)
            for k, v in instance.items()
        }

    def get_model(self, data_type, prefix="") -> Type[BaseModel]:
        kwargs = {
            str(k): (
                v.get_serializer().get_model(prefix + str(k) + "_"),
                ...,
            )
            for k, v in data_type.item_types.items()
        }
        return create_model(prefix + "DictType", **kwargs)  # type: ignore


class DictWriter(DataWriter):
    """Writer for dicts"""

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
        for key, dtype in data.item_types.items():
            dtype_reader, art = dtype.get_writer().write(
                dtype.copy().bind(data.data[key]),
                storage,
                posixpath.join(path, str(key)),
            )
            res[str(key)] = art
            readers[key] = dtype_reader
        return DictReader(data_type=data, item_readers=readers), {
            k: v
            for k, v in dict(flatdict.FlatterDict(res, delimiter="/")).items()
            if v
        }


class DictReader(DataReader[DictType]):
    """Reader for dicts"""

    type: ClassVar[str] = "dict"
    data_type: DictType
    item_readers: Dict[Union[StrictStr, StrictInt], DataReader]
    """Nested readers"""

    def read(self, artifacts: Artifacts) -> DictType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_dict = {}
        for key, dtype_reader in self.item_readers.items():
            v_data_type = dtype_reader.read(artifacts.get(str(key), {}))  # type: ignore[arg-type]
            data_dict[key] = v_data_type.data
        return self.data_type.copy().bind(data_dict)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DictType]:
        raise NotImplementedError


class DynamicDictType(WithDefaultSerializer, DataType):
    """
    Dynamic DataType for dict without fixed set of keys
    """

    type: ClassVar[str] = "d_dict"

    key_type: PrimitiveType
    """DataType for key (primitive)"""
    value_type: DataType
    """DataType for value"""

    @validator("key_type")
    def is_valid_key_type(  # pylint: disable=no-self-argument
        cls, key_type  # noqa: B902
    ):
        if key_type.ptype not in ["str", "int", "float"]:
            raise ValueError(f"key_type {key_type.ptype} is not supported")
        return key_type

    @classmethod
    def process(cls, obj, **kwargs) -> "DynamicDictType":
        return DynamicDictType(
            key_type=DataAnalyzer.analyze(
                next(iter(obj.keys())), is_dynamic=True, **kwargs
            ),
            value_type=DataAnalyzer.analyze(
                next(iter(obj.values())), is_dynamic=True, **kwargs
            ),
        )

    def check_types(self, obj, exc_type, ignore_key_type: bool = False):
        self.check_type(obj, dict, exc_type)

        obj_type = self.process(obj)
        if ignore_key_type:
            obj_types: Union[
                Tuple[PrimitiveType, DataType], Tuple[DataType]
            ] = (obj_type.value_type,)
            expected_types: Union[
                Tuple[PrimitiveType, DataType], Tuple[DataType]
            ] = (self.value_type,)
        else:
            obj_types = (obj_type.key_type, obj_type.value_type)
            expected_types = (self.key_type, self.value_type)
        if obj_types != expected_types:
            raise exc_type(
                f"given dict has type: {obj_types}, expected: {expected_types}"
            )

    def get_requirements(self) -> Requirements:
        return sum(
            [
                self.key_type.get_requirements(),
                self.value_type.get_requirements(),
            ],
            Requirements.new(),
        )

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DynamicDictWriter":
        return DynamicDictWriter(**kwargs)


class DynamicDictSerializer(DataSerializer[DynamicDictType]):
    """Serialzier for dynamic dict"""

    data_class: ClassVar = DynamicDictType
    is_default: ClassVar = True

    def deserialize(self, data_type, obj):
        data_type.check_type(obj, dict, DeserializationError)
        return {
            data_type.key_type.get_serializer()
            .deserialize(k): data_type.value_type.get_serializer()
            .deserialize(
                v,
            )
            for k, v in obj.items()
        }

    def serialize(self, data_type, instance: dict):
        data_type.check_types(instance, SerializationError)

        return {
            data_type.key_type.get_serializer()
            .serialize(k): data_type.value_type.get_serializer()
            .serialize(
                v,
            )
            for k, v in instance.items()
        }

    def get_model(self, data_type, prefix="") -> Type[BaseModel]:
        field_type = (
            Dict[  # type: ignore[index]
                data_type.key_type.get_serializer().get_model(
                    prefix + "_key_"  # noqa: F821  # wtf
                ),
                data_type.value_type.get_serializer().get_model(
                    prefix + "_val_"  # noqa: F821
                ),
            ],
            ...,
        )
        return create_model(prefix + "DynamicDictType", __root__=field_type)  # type: ignore


class DynamicDictWriter(DataWriter):
    """Write dicts without fixed set of keys"""

    type: ClassVar[str] = "d_dict"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        if not isinstance(data, DynamicDictType):
            raise ValueError(
                f"expected data to be of DynamicDictTypeWriter, got {type(data)} instead"
            )
        with storage.open(path) as (f, art):
            f.write(
                json.dumps(data.get_serializer().serialize(data.data)).encode(
                    "utf-8"
                )
            )
        return DynamicDictReader(data_type=data), {DataWriter.art_name: art}


class DynamicDictReader(DataReader[DynamicDictType]):
    """Read dicts without fixed set of keys"""

    type: ClassVar[str] = "d_dict"
    data_type: DynamicDictType

    def read(self, artifacts: Artifacts) -> DynamicDictType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            data = json.load(f)
            # json stores keys as strings. Deserialize string keys as well as values.
            data = self.data_type.get_serializer().deserialize(data)
        return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DynamicDictType]:
        raise NotImplementedError


class BinaryDataType(
    WithDefaultSerializer, DataType, DataHook, IsInstanceHookMixin
):
    """Represents raw binary data"""

    type: ClassVar = "binary"

    valid_types: ClassVar = (bytes,)

    @classmethod
    def process(cls, obj: Any, **kwargs) -> DataType:
        return BinaryDataType().bind(obj)

    def get_requirements(self) -> Requirements:
        return Requirements.new()

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> "DataWriter":
        return BinaryDataWriter()


class BinaryDataSerializer(BinarySerializer[BinaryDataType]):
    """Serializer for binary data"""

    data_class: ClassVar = BinaryDataType
    is_default: ClassVar = True

    def serialize(self, data_type, instance: bytes) -> bytes:
        return instance

    @contextlib.contextmanager
    def dump(self, data_type, instance: Any) -> Iterator[BinaryIO]:
        yield io.BytesIO(instance)

    def deserialize(self, data_type, obj: Union[bytes, BinaryIO]) -> Any:
        if isinstance(obj, bytes):
            return obj
        return obj.read()


class BinaryDataWriter(DataWriter[BinaryDataType]):
    """Writer for binary data"""

    type: ClassVar = "binary"

    def write(
        self, data: BinaryDataType, storage: Storage, path: str
    ) -> Tuple[DataReader[BinaryDataType], Artifacts]:
        with storage.open(path) as (f, art):
            f.write(data.data)
        return BinaryDataReader(data_type=data), {self.art_name: art}


class BinaryDataReader(DataReader[BinaryDataType]):
    """Reader for binary data"""

    type: ClassVar = "binary"

    def read(self, artifacts: Artifacts) -> BinaryDataType:
        art = artifacts[BinaryDataWriter.art_name]
        with art.open() as f:
            return BinaryDataType().bind(f.read())

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[BinaryDataType]:
        raise NotImplementedError


class FileSerializer(BinarySerializer):
    """BinarySerialzier for arbitrary data using reader and writer"""

    type: ClassVar = "file"
    support_files: ClassVar = True

    @staticmethod
    def _get_artifact(data_type: DataType, instance: Any) -> InMemoryArtifact:
        writer = data_type.get_writer()
        _, arts = writer.write(data_type.bind(instance), InMemoryStorage(), "")
        art = arts[writer.art_name]
        assert isinstance(art, InMemoryArtifact)  # TODO make buffered
        return art

    def serialize(self, data_type, instance: Any) -> bytes:
        return self._get_artifact(data_type, instance).payload

    @contextlib.contextmanager
    def dump(self, data_type, instance: Any) -> Iterator[BinaryIO]:
        with self._get_artifact(data_type, instance).open() as f:
            yield f

    def deserialize(self, data_type, obj: Union[bytes, BinaryIO]) -> Any:
        writer = data_type.get_writer()
        reader: DataReader = writer.get_reader(data_type)
        art: Artifact
        if isinstance(obj, bytes):
            art = InMemoryArtifact(uri="", size=-1, hash="", payload=obj)
        else:
            art = InMemoryFileobjArtifact(
                uri="", size=-1, hash="", fileobj=obj
            )
        return reader.read({writer.art_name: art}).data
