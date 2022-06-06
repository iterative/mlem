import pytest
from pydantic import parse_obj_as

from mlem.core.data_type import (
    ArrayReader,
    ArrayType,
    DataAnalyzer,
    DataReader,
    DataType,
    DictReader,
    DictType,
    ListType,
    PrimitiveReader,
    PrimitiveType,
    TupleType,
    _TupleLikeReader,
    _TupleLikeWriter,
)
from tests.conftest import data_write_read_check


class NotPrimitive:
    pass


def test_primitives_not_ok():
    assert not PrimitiveType.is_object_valid(NotPrimitive())


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


def test_array():
    l_value = [1, 2, 3, 4, 5]
    dt = DataAnalyzer.analyze(l_value)
    assert isinstance(dt, ArrayType)
    payload = {
        "dtype": {"ptype": "int", "type": "primitive"},
        "size": 5,
        "type": "array",
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(ArrayType, payload)
    assert dt2 == dt
    assert l_value == dt.serialize(l_value)
    assert l_value == dt.deserialize(l_value)
    assert dt.get_model().__name__ == "Array"
    assert dt.get_model().schema() == {
        "items": {"type": "integer"},
        "title": "Array",
        "type": "array",
    }


def test_list_source():
    l_value = [1, 2, 3, 4, 5]
    dt = DataType.create(l_value)

    artifacts = data_write_read_check(
        dt,
        reader_type=ArrayReader,
    )

    assert list(artifacts.keys()) == [f"{x}/data" for x in range(len(l_value))]
    assert artifacts["0/data"].uri.endswith("data/0")
    assert artifacts["1/data"].uri.endswith("data/1")
    assert artifacts["2/data"].uri.endswith("data/2")
    assert artifacts["3/data"].uri.endswith("data/3")
    assert artifacts["4/data"].uri.endswith("data/4")


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


def test_dict():
    d = {"1": 1, "2": "a"}
    dt = DataAnalyzer.analyze(d)
    assert isinstance(dt, DictType)
    payload = {
        "item_types": {
            "1": {"ptype": "int", "type": "primitive"},
            "2": {"ptype": "str", "type": "primitive"},
        },
        "type": "dict",
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(DictType, payload)
    assert dt2 == dt
    assert d == dt.serialize(d)
    assert d == dt.deserialize(d)
    assert dt.get_model().__name__ == "DictType"
    assert dt.get_model().schema() == {
        "title": "DictType",
        "type": "object",
        "properties": {
            "1": {"title": "1", "type": "integer"},
            "2": {"title": "2", "type": "string"},
        },
        "required": ["1", "2"],
    }


def test_dict_source():
    d_value = {"1": 1.5, "2": "a", "3": {"1": False}}
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
