"""
Base classes for working with data in MLEM
"""
import builtins
import json
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
from pydantic import BaseModel, StrictInt, StrictStr, validator
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
    def create(cls, obj: Any, is_dynamic: bool = False, **kwargs):
        return DataAnalyzer.analyze(obj, is_dynamic=is_dynamic, **kwargs).bind(
            obj
        )


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
    """Resulting data type"""
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
    """Name of builtin type"""

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
    """Writer for primitive types"""

    type: ClassVar[str] = "primitive"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            f.write(str(data.data).encode("utf-8"))
        return PrimitiveReader(data_type=data), {self.art_name: art}


class PrimitiveReader(DataReader):
    """Reader for primitive types"""

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
    """DataType of elements"""
    size: Optional[int]
    """Size of the list"""

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


class _TupleLikeType(DataType, DataSerializer):
    """
    DataType for tuple-like collections
    """

    type: ClassVar = "_tuple_like"
    actual_type: ClassVar[type]

    items: List[DataType]
    """DataTypes of elements"""

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


class DictType(DataType, DataSerializer):
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
            str(k): (v.get_serializer().get_model(prefix + str(k) + "_"), ...)
            for k, v in self.item_types.items()
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
        for (key, dtype) in data.item_types.items():
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


class DictReader(DataReader):
    """Reader for dicts"""

    type: ClassVar[str] = "dict"
    data_type: DictType
    item_readers: Dict[Union[StrictStr, StrictInt], DataReader]
    """Nested readers"""

    def read(self, artifacts: Artifacts) -> DataType:
        artifacts = flatdict.FlatterDict(artifacts, delimiter="/")
        data_dict = {}
        for (key, dtype_reader) in self.item_readers.items():
            v_data_type = dtype_reader.read(artifacts.get(str(key), {}))  # type: ignore[arg-type]
            data_dict[key] = v_data_type.data
        return self.data_type.copy().bind(data_dict)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class DynamicDictType(DataType, DataSerializer):
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

    def deserialize(self, obj):
        self.check_type(obj, dict, DeserializationError)
        return {
            self.key_type.get_serializer()
            .deserialize(
                k,
            ): self.value_type.get_serializer()
            .deserialize(
                v,
            )
            for k, v in obj.items()
        }

    def serialize(self, instance: dict):
        self._check_types(instance, SerializationError)

        return {
            self.key_type.get_serializer()
            .serialize(
                k,
            ): self.value_type.get_serializer()
            .serialize(
                v,
            )
            for k, v in instance.items()
        }

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

    def _check_types(self, obj, exc_type, ignore_key_type: bool = False):
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

    def get_model(self, prefix="") -> Type[BaseModel]:
        field_type = (
            Dict[  # type: ignore
                self.key_type.get_serializer().get_model(
                    prefix + "_key_"  # noqa: F821
                ),
                self.value_type.get_serializer().get_model(
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


class DynamicDictReader(DataReader):
    """Read dicts without fixed set of keys"""

    type: ClassVar[str] = "d_dict"
    data_type: DynamicDictType

    def read(self, artifacts: Artifacts) -> DataType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            data = json.load(f)
            # json stores keys as strings. Deserialize string keys as well as values.
            data = self.data_type.deserialize(data)
        return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError
