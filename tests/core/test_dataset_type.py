import pytest
from pydantic import parse_obj_as

from mlem.core.dataset_type import (
    DatasetAnalyzer,
    DatasetType,
    DictDatasetType,
    DictReader,
    ListDatasetType,
    ListReader,
    PrimitiveReader,
    PrimitiveType,
    TupleDatasetType,
    _TupleLikeDatasetReader,
    _TupleLikeDatasetWriter,
)
from tests.conftest import dataset_write_read_check


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
    dataset = DatasetType.create(data)

    def custom_assert(x, y):
        assert x == y
        assert isinstance(x, ptype)
        assert isinstance(y, ptype)

    dataset_write_read_check(
        dataset,
        reader_type=PrimitiveReader,
        custom_assert=custom_assert,
    )


@pytest.mark.parametrize("ptype", PrimitiveType.PRIMITIVES - {complex})
def test_primitives(ptype):
    value = ptype()
    assert PrimitiveType.is_object_valid(value)
    dt = DatasetAnalyzer.analyze(value)
    assert isinstance(dt, PrimitiveType)
    assert dt.ptype == ptype.__name__
    payload = {"ptype": ptype.__name__, "type": "primitive"}
    assert dt.dict() == payload
    dt2 = parse_obj_as(DatasetType, payload)
    assert isinstance(dt2, PrimitiveType)
    assert dt2 == dt
    assert dt2.to_type == ptype
    assert dt.get_model() is ptype


def test_list():
    l_value = [1, 2, 3, 4, 5]
    dt = DatasetAnalyzer.analyze(l_value)
    assert isinstance(dt, ListDatasetType)
    payload = {
        "dtype": {"ptype": "int", "type": "primitive"},
        "size": 5,
        "type": "list",
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(ListDatasetType, payload)
    assert dt2 == dt
    assert l_value == dt.serialize(l_value)
    assert l_value == dt.deserialize(l_value)
    assert dt.get_model().__name__ == "ListDataset"
    assert dt.get_model().schema() == {
        "items": {"type": "integer"},
        "title": "ListDataset",
        "type": "array",
    }


def test_list_source():
    l_value = [1, 2, 3, 4, 5]
    dt = DatasetType.create(l_value)

    artifacts = dataset_write_read_check(
        dt,
        reader_type=ListReader,
    )

    assert list(artifacts.keys()) == list(
        map(lambda x: str(x) + "/data", range(len(l_value)))
    )
    assert artifacts["0"]["data"].uri.endswith("data/0")
    assert artifacts["1"]["data"].uri.endswith("data/1")
    assert artifacts["2"]["data"].uri.endswith("data/2")
    assert artifacts["3"]["data"].uri.endswith("data/3")
    assert artifacts["4"]["data"].uri.endswith("data/4")


def test_tuple():
    t = (1, 2, 3)
    dt = DatasetAnalyzer.analyze(t)
    assert isinstance(dt, TupleDatasetType)
    payload = {
        "items": [
            {"ptype": "int", "type": "primitive"},
            {"ptype": "int", "type": "primitive"},
            {"ptype": "int", "type": "primitive"},
        ],
        "type": "tuple",
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(TupleDatasetType, payload)
    assert dt2 == dt
    assert t == dt.serialize(t)
    assert t == dt.deserialize(t)
    assert dt.get_model().__name__ == "_TupleLikeDataset"
    assert dt.get_model().schema() == {
        "title": "_TupleLikeDataset",
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
    dt = DatasetType.create(t_value)

    artifacts = dataset_write_read_check(
        dt,
        reader_type=_TupleLikeDatasetReader,
        writer=_TupleLikeDatasetWriter(),
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
    assert list(artifacts["1"].keys()) == list(
        map(lambda x: str(x) + "/data", range(len(t_value[1])))
    )
    assert artifacts["0"]["data"].uri.endswith("data/0")
    assert artifacts["1"]["0"]["data"].uri.endswith("data/1/0")
    assert artifacts["1"]["1"]["data"].uri.endswith("data/1/1")
    assert artifacts["2"]["data"].uri.endswith("data/2")
    assert artifacts["3"]["data"].uri.endswith("data/3")
    assert artifacts["4"]["data"].uri.endswith("data/4")
    assert artifacts["5"]["data"].uri.endswith("data/5")


def test_mixed_list_source():
    t_value = [1, [3, 7], False, 3.2, "mlem", None]
    dt = DatasetType.create(t_value)

    artifacts = dataset_write_read_check(
        dt,
        reader_type=_TupleLikeDatasetReader,
        writer=_TupleLikeDatasetWriter(),
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
    assert list(artifacts["1"].keys()) == list(
        map(lambda x: str(x) + "/data", range(len(t_value[1])))
    )
    assert artifacts["0"]["data"].uri.endswith("data/0")
    assert artifacts["1"]["0"]["data"].uri.endswith("data/1/0")
    assert artifacts["1"]["1"]["data"].uri.endswith("data/1/1")
    assert artifacts["2"]["data"].uri.endswith("data/2")
    assert artifacts["3"]["data"].uri.endswith("data/3")
    assert artifacts["4"]["data"].uri.endswith("data/4")
    assert artifacts["5"]["data"].uri.endswith("data/5")


def test_dict():
    d = {"1": 1, "2": "a"}
    dt = DatasetAnalyzer.analyze(d)
    assert isinstance(dt, DictDatasetType)
    payload = {
        "item_types": {
            "1": {"ptype": "int", "type": "primitive"},
            "2": {"ptype": "str", "type": "primitive"},
        },
        "type": "dict",
    }
    assert dt.dict() == payload
    dt2 = parse_obj_as(DictDatasetType, payload)
    assert dt2 == dt
    assert d == dt.serialize(d)
    assert d == dt.deserialize(d)
    assert dt.get_model().__name__ == "DictDataset"
    assert dt.get_model().schema() == {
        "title": "DictDataset",
        "type": "object",
        "properties": {
            "1": {"title": "1", "type": "integer"},
            "2": {"title": "2", "type": "string"},
        },
        "required": ["1", "2"],
    }


def test_dict_source():
    d_value = {"1": 1.5, "2": "a", "3": {"1": False}}
    dataset = DatasetType.create(d_value)

    def custom_assert(x, y):
        assert x == y
        assert len(x) == len(y)
        assert isinstance(x, dict)
        assert isinstance(y, dict)

    artifacts = dataset_write_read_check(
        dataset,
        reader_type=DictReader,
        custom_assert=custom_assert,
    )

    assert list(artifacts.keys()) == ["1/data", "2/data", "3/1/data"]
    assert list(artifacts["3"].keys()) == ["1/data"]
    assert artifacts["1"]["data"].uri.endswith("data/1")
    assert artifacts["2"]["data"].uri.endswith("data/2")
    assert artifacts["3"]["1"]["data"].uri.endswith("data/3/1")
