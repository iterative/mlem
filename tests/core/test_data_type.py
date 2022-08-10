import copy

import numpy as np
import pytest
from pydantic import parse_obj_as
from pytest_lazyfixture import lazy_fixture

from mlem.core.artifacts import Artifact
from mlem.core.data_type import (
    ArrayReader,
    ArrayType,
    DataAnalyzer,
    DataReader,
    DataType,
    DictReader,
    DictType,
    DynamicDictReader,
    DynamicDictType,
    ListType,
    PrimitiveReader,
    PrimitiveType,
    TupleType,
    _TupleLikeReader,
    _TupleLikeWriter,
)
from tests.conftest import data_write_read_check


@pytest.fixture
def dict_data_str_keys():
    data = {"1": 1.5, "2": "a", "3": {"1": False}}
    payload = {
        "item_types": {
            "1": {"ptype": "float", "type": "primitive"},
            "2": {"ptype": "str", "type": "primitive"},
            "3": {
                "item_types": {"1": {"ptype": "bool", "type": "primitive"}},
                "type": "dict",
            },
        },
        "type": "dict",
    }
    schema = {
        "title": "DictType",
        "type": "object",
        "properties": {
            "1": {"title": "1", "type": "number"},
            "2": {"title": "2", "type": "string"},
            "3": {"$ref": "#/definitions/3_DictType"},
        },
        "required": ["1", "2", "3"],
        "definitions": {
            "3_DictType": {
                "title": "3_DictType",
                "type": "object",
                "properties": {"1": {"title": "1", "type": "boolean"}},
                "required": ["1"],
            }
        },
    }

    return data, payload, schema


@pytest.fixture
def dict_data_int_keys(dict_data_str_keys):
    data = {"1": 1.5, 2: "a", "3": {1: False}}
    payload = {
        "item_types": {
            "1": {"ptype": "float", "type": "primitive"},
            2: {"ptype": "str", "type": "primitive"},
            "3": {
                "item_types": {1: {"ptype": "bool", "type": "primitive"}},
                "type": "dict",
            },
        },
        "type": "dict",
    }
    return data, payload, dict_data_str_keys[2]


class NotPrimitive:
    pass


def test_primitives_not_ok():
    assert not PrimitiveType.is_object_valid(NotPrimitive())


@pytest.fixture
def array():
    is_dynamic = False
    array = [1, 2, 3, 4, 5]
    payload = {
        "dtype": {"ptype": "int", "type": "primitive"},
        "size": 5,
        "type": "array",
    }
    schema = {
        "items": {"type": "integer"},
        "title": "Array",
        "type": "array",
    }

    return is_dynamic, array, payload, schema


@pytest.fixture
def array_dynamic(array):
    is_dynamic = True
    payload = copy.deepcopy(array[2])
    del payload["size"]
    return is_dynamic, array[1], payload, array[3]


@pytest.mark.parametrize("ptype", PrimitiveType.PRIMITIVES)
def test_primitive_source(ptype):
    if ptype is type(None):  # noqa: E721
        data = None
    else:
        data = ptype(1.5)
    data_type = DataType.create(data)

    def custom_assert(x, y):
        assert x == y
        assert isinstance(x, ptype)
        assert isinstance(y, ptype)

    data_write_read_check(
        data_type,
        reader_type=PrimitiveReader,
        custom_assert=custom_assert,
    )


@pytest.mark.parametrize("ptype", PrimitiveType.PRIMITIVES - {complex})
def test_primitives(ptype):
    value = ptype()
    assert PrimitiveType.is_object_valid(value)
    dt = DataAnalyzer.analyze(value)
    assert isinstance(dt, PrimitiveType)
    assert dt.ptype == ptype.__name__
    payload = {"ptype": ptype.__name__, "type": "primitive"}
    assert dt.dict() == payload
    dt2 = parse_obj_as(DataType, payload)
    assert isinstance(dt2, PrimitiveType)
    assert dt2 == dt
    assert dt2.to_type == ptype
    assert dt.get_model() is ptype


@pytest.mark.parametrize(
    "array_data,value",
    [
        (lazy_fixture("array"), None),
        (lazy_fixture("array_dynamic"), None),
        (lazy_fixture("array_dynamic"), [1, 2, 3]),
    ],
)
def test_array(array_data, value):
    dt = DataAnalyzer.analyze(array_data[1], is_dynamic=array_data[0])
    l_value = array_data[1] if value is None else value
    assert isinstance(dt, ArrayType)
    assert dt.dict() == array_data[2]
    dt2 = parse_obj_as(ArrayType, array_data[2])
    assert dt2 == dt
    assert l_value == dt.serialize(l_value)
    assert l_value == dt.deserialize(l_value)
    assert dt.get_model().__name__ == "Array"
    assert dt.get_model().schema() == array_data[3]


@pytest.mark.parametrize(
    "array_data,value",
    [
        (lazy_fixture("array"), None),
        (lazy_fixture("array_dynamic"), None),
        (lazy_fixture("array_dynamic"), [1, 2, 3]),
    ],
)
def test_list_source(array_data, value):
    dt = DataType.create(array_data[1])
    l_value = array_data[1] if value is None else value
    dt.bind(l_value)

    artifacts = data_write_read_check(
        dt,
        reader_type=ArrayReader,
    )

    assert list(artifacts.keys()) == [f"{x}/data" for x in range(len(l_value))]
    for x in range(len(l_value)):
        assert artifacts[f"{x}/data"].uri.endswith(f"data/{x}")


def test_tuple():
    t = (1, 2, 3)
    dt = DataAnalyzer.analyze(t)
    assert isinstance(dt, TupleType)
    payload = {
        "items": [
            {"ptype": "int", "type": "primitive"},
            {"ptype": "int", "type": "primitive"},
            {"ptype": "int", "type": "primitive"},
        ],
        "type": "tuple",
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(TupleType, payload)
    assert dt2 == dt
    assert t == dt.serialize(t)
    assert t == dt.deserialize(t)
    assert dt.get_model().__name__ == "_TupleLikeType"
    assert dt.get_model().schema() == {
        "title": "_TupleLikeType",
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": [
            {"type": "integer"},
            {"type": "integer"},
            {"type": "integer"},
        ],
    }


def test_tuple_source():
    t_value = (1, [3, 7], False, 3.2, "mlem", None)
    dt = DataType.create(t_value)

    artifacts = data_write_read_check(
        dt,
        reader_type=_TupleLikeReader,
        writer=_TupleLikeWriter(),
    )

    assert list(artifacts.keys()) == [
        "0/data",
        "1/0/data",
        "1/1/data",
        "2/data",
        "3/data",
        "4/data",
        "5/data",
    ]
    assert artifacts["0/data"].uri.endswith("data/0")
    assert artifacts["1/0/data"].uri.endswith("data/1/0")
    assert artifacts["1/1/data"].uri.endswith("data/1/1")
    assert artifacts["2/data"].uri.endswith("data/2")
    assert artifacts["3/data"].uri.endswith("data/3")
    assert artifacts["4/data"].uri.endswith("data/4")
    assert artifacts["5/data"].uri.endswith("data/5")


def test_list_reader():
    data_type = ListType(items=[])
    assert data_type.dict()["type"] == "list"
    reader = _TupleLikeReader(data_type=data_type, readers=[])
    new_reader = parse_obj_as(DataReader, reader.dict())
    res = new_reader.read({})
    assert res.data == []


def test_mixed_list_source():
    t_value = [1, [3, 7], False, 3.2, "mlem", None]
    dt = DataType.create(t_value)

    artifacts = data_write_read_check(
        dt,
        reader_type=_TupleLikeReader,
        writer=_TupleLikeWriter(),
    )

    assert list(artifacts.keys()) == [
        "0/data",
        "1/0/data",
        "1/1/data",
        "2/data",
        "3/data",
        "4/data",
        "5/data",
    ]
    assert artifacts["0/data"].uri.endswith("data/0")
    assert artifacts["1/0/data"].uri.endswith("data/1/0")
    assert artifacts["1/1/data"].uri.endswith("data/1/1")
    assert artifacts["2/data"].uri.endswith("data/2")
    assert artifacts["3/data"].uri.endswith("data/3")
    assert artifacts["4/data"].uri.endswith("data/4")
    assert artifacts["5/data"].uri.endswith("data/5")


@pytest.fixture
def dict_data():
    is_dynamic = False
    d = {"1": 1, "2": "a"}
    payload = {
        "item_types": {
            "1": {"ptype": "int", "type": "primitive"},
            "2": {"ptype": "str", "type": "primitive"},
        },
        "type": "dict",
    }

    schema = {
        "title": "DictType",
        "type": "object",
        "properties": {
            "1": {"title": "1", "type": "integer"},
            "2": {"title": "2", "type": "string"},
        },
        "required": ["1", "2"],
    }

    test_data1 = {"1": 1, "2": "a"}
    test_data2 = {"1": 2, "2": "b"}
    test_data3 = {"1": 3, "2": "c"}

    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_data():
    is_dynamic = True
    d = {"a": 1, "b": 2}
    payload = {
        "key_type": {"ptype": "str", "type": "primitive"},
        "value_type": {"ptype": "int", "type": "primitive"},
        "type": "d_dict",
    }
    schema = {
        "title": "DynamicDictType",
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }

    test_data1 = {"a": 1, "b": 2}
    test_data2 = {"a": 1}
    test_data3 = {"a": 1, "b": 2, "c": 3, "d": 1}

    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_str_val_type_data():
    is_dynamic = True
    d = {"a": "1", "b": "2"}
    payload = {
        "key_type": {"ptype": "str", "type": "primitive"},
        "value_type": {"ptype": "str", "type": "primitive"},
        "type": "d_dict",
    }
    schema = {
        "title": "DynamicDictType",
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    test_data1 = {"a": "1", "b": "2"}
    test_data2 = {"a": "1"}
    test_data3 = {"a": "1", "b": "2", "c": "3", "d": "1"}

    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_int_key_type_data():
    is_dynamic = True
    d = {1: "1", 2: "2"}
    payload = {
        "key_type": {"ptype": "int", "type": "primitive"},
        "value_type": {"ptype": "str", "type": "primitive"},
        "type": "d_dict",
    }
    schema = {
        "title": "DynamicDictType",
        "type": "object",
        "additionalProperties": {"type": "string"},
    }

    test_data1 = {1: "1", 2: "2"}
    test_data2 = {2: "1"}
    test_data3 = {3: "1", 4: "2", 5: "3", 6: "1"}

    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_float_key_type_data():
    is_dynamic = True
    d = {1.0: "1", 2.0: "2"}
    payload = {
        "key_type": {"ptype": "float", "type": "primitive"},
        "value_type": {"ptype": "str", "type": "primitive"},
        "type": "d_dict",
    }
    schema = {
        "title": "DynamicDictType",
        "type": "object",
        "additionalProperties": {"type": "string"},
    }

    test_data1 = {1.0: "1", 2.0: "2"}
    test_data2 = {2.9999999999: "1"}
    test_data3 = {3.8998: "1", 4.0001: "2", 5.2: "3", 6.9: "1"}

    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_array_type():
    is_dynamic = True
    d = {"a": [1, 2, 3], "b": [3, 4, 5]}
    payload = {
        "key_type": {"ptype": "str", "type": "primitive"},
        "type": "d_dict",
        "value_type": {
            "dtype": {"ptype": "int", "type": "primitive"},
            "type": "array",
        },
    }
    schema = {
        "additionalProperties": {"$ref": "#/definitions/_val_Array"},
        "definitions": {
            "_val_Array": {
                "items": {"type": "integer"},
                "title": "_val_Array",
                "type": "array",
            }
        },
        "title": "DynamicDictType",
        "type": "object",
    }

    test_data1 = {"a": [1, 2, 3], "b": [3, 4, 5]}
    test_data2 = {"a": [1, 2, 3]}
    test_data3 = {"a": [1, 2, 3], "b": [3, 4, 5], "d": [6, 7, 8]}
    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_dict_type():
    is_dynamic = True
    d = {"a": {"l": [1, 2]}, "b": {"l": [3, 4]}}
    payload = {
        "key_type": {"ptype": "str", "type": "primitive"},
        "type": "d_dict",
        "value_type": {
            "key_type": {"ptype": "str", "type": "primitive"},
            "type": "d_dict",
            "value_type": {
                "dtype": {"ptype": "int", "type": "primitive"},
                "type": "array",
            },
        },
    }
    schema = {
        "additionalProperties": {"$ref": "#/definitions/_val_DynamicDictType"},
        "definitions": {
            "_val_DynamicDictType": {
                "additionalProperties": {
                    "$ref": "#/definitions/_val__val_Array"
                },
                "title": "_val_DynamicDictType",
                "type": "object",
            },
            "_val__val_Array": {
                "items": {"type": "integer"},
                "title": "_val__val_Array",
                "type": "array",
            },
        },
        "title": "DynamicDictType",
        "type": "object",
    }
    test_data1 = {"a": {"l": [1, 2]}, "b": {"l": [3, 4]}}
    test_data2 = {"a": {"l": [1, 2]}}
    test_data3 = {"a": {"l": [1, 2]}, "b": {"l": [3, 4]}, "c": {"k": [3, 4]}}
    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.fixture
def dynamic_dict_ndarray_type(numpy_default_int_dtype):
    is_dynamic = True
    d = {11: {1: np.array([1, 2])}, 22: {2: np.array([3, 4])}}
    payload = {
        "key_type": {"ptype": "int", "type": "primitive"},
        "type": "d_dict",
        "value_type": {
            "key_type": {"ptype": "int", "type": "primitive"},
            "type": "d_dict",
            "value_type": {
                "dtype": numpy_default_int_dtype,
                "shape": (None,),
                "type": "ndarray",
            },
        },
    }
    schema = {
        "additionalProperties": {"$ref": "#/definitions/_val_DynamicDictType"},
        "definitions": {
            "_val_DynamicDictType": {
                "additionalProperties": {
                    "$ref": "#/definitions/_val__val_NumpyNdarray"
                },
                "title": "_val_DynamicDictType",
                "type": "object",
            },
            "_val__val_NumpyNdarray": {
                "items": {"type": "integer"},
                "title": "_val__val_NumpyNdarray",
                "type": "array",
            },
        },
        "title": "DynamicDictType",
        "type": "object",
    }

    test_data1 = {11: {1: np.array([1, 2])}, 22: {2: np.array([3, 4])}}
    test_data2 = {11: {1: np.array([1, 2])}}
    test_data3 = {
        11: {1: np.array([1, 2])},
        22: {2: np.array([3, 4])},
        33: {2: np.array([5, 6])},
    }
    return is_dynamic, d, payload, schema, test_data1, test_data2, test_data3


@pytest.mark.parametrize("test_data_idx", [4, 5, 6])
@pytest.mark.parametrize(
    "data",
    [
        lazy_fixture("dict_data"),
        lazy_fixture("dynamic_dict_data"),
        lazy_fixture("dynamic_dict_str_val_type_data"),
        lazy_fixture("dynamic_dict_int_key_type_data"),
        lazy_fixture("dynamic_dict_float_key_type_data"),
        lazy_fixture("dynamic_dict_array_type"),
        lazy_fixture("dynamic_dict_dict_type"),
        lazy_fixture("dynamic_dict_ndarray_type"),
    ],
)
def test_dict(data, test_data_idx):
    is_dynamic, d, payload, schema, test_data = (
        data[0],
        data[1],
        data[2],
        data[3],
        data[test_data_idx],
    )
    dt = DataAnalyzer.analyze(d, is_dynamic=is_dynamic)
    dtype = DictType if not is_dynamic else DynamicDictType
    assert isinstance(dt, dtype)

    assert dt.dict() == payload
    dt2 = parse_obj_as(dtype, payload)
    assert dt2 == dt
    serialised_test_data = dt.serialize(test_data)
    deserialised_test_data = dt.deserialize(serialised_test_data)
    assert serialised_test_data == dt.serialize(deserialised_test_data)
    assert dt.get_model().__name__ == dtype.__name__
    assert dt.get_model().schema() == schema
    assert parse_obj_as(dt.get_model(), serialised_test_data)


@pytest.mark.parametrize("test_data_idx", [4, 5, 6])
@pytest.mark.parametrize(
    "data",
    [
        lazy_fixture("dict_data"),
        lazy_fixture("dynamic_dict_data"),
        lazy_fixture("dynamic_dict_str_val_type_data"),
        lazy_fixture("dynamic_dict_int_key_type_data"),
        lazy_fixture("dynamic_dict_float_key_type_data"),
        lazy_fixture("dynamic_dict_array_type"),
        lazy_fixture("dynamic_dict_dict_type"),
        lazy_fixture("dynamic_dict_ndarray_type"),
    ],
)
def test_dict_source(data, test_data_idx):
    is_dynamic, d, test_data = (
        data[0],
        data[1],
        data[test_data_idx],
    )
    data_type = DataType.create(d, is_dynamic=is_dynamic)
    data_type = data_type.bind(test_data)
    dtype_reader = DynamicDictReader if is_dynamic else DictReader

    def custom_assert(x, y):
        np.testing.assert_equal(x, y)
        assert len(x) == len(y)
        assert isinstance(x, dict)
        assert isinstance(y, dict)

    artifacts = data_write_read_check(
        data_type,
        reader_type=dtype_reader,
        custom_assert=custom_assert,
    )

    if not is_dynamic:
        assert list(artifacts.keys()) == ["1/data", "2/data"]
        assert artifacts["1/data"].uri.endswith("data/1")
        assert artifacts["2/data"].uri.endswith("data/2")
    else:
        assert list(artifacts.keys()) == ["data"]
        assert artifacts["data"].uri.endswith("data")


@pytest.mark.parametrize(
    "dict_data",
    [lazy_fixture("dict_data_str_keys"), lazy_fixture("dict_data_int_keys")],
)
def test_dict_key_int_and_str_types(dict_data):
    d_value, payload, schema = dict_data
    data_type = DataType.create(d_value)

    assert isinstance(data_type, DictType)

    assert data_type.dict() == payload
    dt2 = parse_obj_as(DictType, payload)
    assert dt2 == data_type
    assert d_value == data_type.serialize(d_value)
    assert d_value == data_type.deserialize(d_value)
    assert data_type.get_model().__name__ == "DictType"
    assert data_type.get_model().schema() == schema


@pytest.mark.parametrize(
    "d_value",
    [
        {"1": 1.5, "2": "a", "3": {"1": False}},
        {"1": 1.5, 2: "a", "3": {1: False}},
    ],
)
def test_dict_source_int_and_str_types(d_value):
    data_type = DataType.create(d_value)

    def custom_assert(x, y):
        assert x == y
        assert len(x) == len(y)
        assert isinstance(x, dict)
        assert isinstance(y, dict)

    artifacts = data_write_read_check(
        data_type,
        reader_type=DictReader,
        custom_assert=custom_assert,
    )

    assert list(artifacts.keys()) == ["1/data", "2/data", "3/1/data"]
    assert artifacts["1/data"].uri.endswith("data/1")
    assert artifacts["2/data"].uri.endswith("data/2")
    assert artifacts["3/1/data"].uri.endswith("data/3/1")


def test_empty_nested_dict():
    data = {"a": 1, "b": {}}

    data_type = DataType.create(data)

    artifacts = data_write_read_check(
        data_type,
        reader_type=DictReader,
    )

    assert all(isinstance(v, Artifact) for v in artifacts.values()), {
        k: type(v) for k, v in artifacts.items()
    }
    assert list(artifacts.keys()) == ["a/data"]
    assert artifacts["a/data"].uri.endswith("data/a")
