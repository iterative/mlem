import pytest
from pydantic import parse_obj_as

from mlem.core.dataset_type import (
    DatasetAnalyzer,
    DatasetType,
    DictDatasetType,
    ListDatasetType,
    PrimitiveType,
    TupleDatasetType,
)

type_schema_map = {
    int: "integer",
    str: "string",
    bool: "boolean",
    float: "number",
    type(None): "null",
}


class NotPrimitive:
    pass


def test_primitives_not_ok():
    assert not PrimitiveType.is_object_valid(NotPrimitive())


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
    assert dt.get_model().__name__ == "Primitive"
    assert dt.get_model().schema() == {
        "title": "Primitive",
        "type": type_schema_map[ptype],
    }


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
        "title": "ListDataset",
        "type": "array",
        "items": {"$ref": "#/definitions/Primitive"},
        "definitions": {
            "Primitive": {"title": "Primitive", "type": "integer"}
        },
    }


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
    # assert dt.get_model().schema() fails due to KeyError: <class 'pydantic.main.Primitive'>, TODO https://github.com/iterative/mlem/issues/194


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
    # assert dt.get_model().schema() fails due to KeyError: <class 'pydantic.main.Primitive'>, TODO https://github.com/iterative/mlem/issues/194
