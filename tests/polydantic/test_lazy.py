from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel, ValidationError, parse_obj_as, validator

from mlem.polydantic.lazy import LazyModel, lazy_field


class Payload(BaseModel):
    value: int

    @validator("value")
    def counter(cls, value):  # pylint: disable=no-self-argument  # noqa: B902
        return value + 1


class Model(BaseModel):
    field_cache: Any
    field, field_raw, field_cache = lazy_field(
        Payload, "field", "field_cache", ...
    )


def test_deserialization_and_cache():
    payload = {"field": {"value": 1}}

    obj = parse_obj_as(Model, payload)

    assert isinstance(obj, Model)
    assert isinstance(obj.field_raw, dict)
    assert isinstance(obj.field, Payload)
    assert obj.field.value == 2
    assert obj.field.value == 2


def test_laziness():

    payload = {"field": {"value": "string"}}

    obj = parse_obj_as(Model, payload)

    assert isinstance(obj, Model)
    assert isinstance(obj.field_raw, dict)
    with pytest.raises(ValidationError):
        print(obj.field)


def test_serialization():
    obj = Model(field=Payload(value=0))

    payload = obj.dict(by_alias=True)
    assert payload == {"field": {"value": 1}}
    assert isinstance(obj.field, Payload)
    assert isinstance(obj.field_raw, dict)


def test_setting_value():
    obj = Model(field=Payload(value=0))

    obj.field.value = 2
    assert obj.field.value == 2
    assert obj.field_raw["value"] == 2


class ModelWithOptional(LazyModel):
    field_cache: Optional[Dict]
    field, field_raw, field_cache = lazy_field(
        Payload,
        "field",
        "field_cache",
        parse_as_type=Optional[Payload],
        default=None,
    )


def test_setting_optional_field():
    obj = ModelWithOptional()
    assert obj.field is None
    obj.field = Payload(value=0)
    assert obj.field.value == 1
    obj.field_raw = {"value": 5}
    assert obj.field.value == 6
