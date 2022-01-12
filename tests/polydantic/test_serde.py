from typing import Any, ClassVar, Optional

from pydantic import BaseModel, parse_obj_as, validator

from mlem.polydantic import PolyModel
from mlem.polydantic.lazy import lazy_field


def test_poly_model_dict():
    class Parent(PolyModel):
        class Config:
            type_root = True
            type_field = "type1"

    class ChildA(Parent):
        type1: ClassVar[str] = "a"
        a: int

    class ChildB(Parent):
        type1 = "b"
        b: str

    a = ChildA(a=1)
    payload = {"type1": "a", "a": 1}
    assert a.dict() == payload
    assert dict(a) == payload

    b = ChildB(b="b")
    payload = {"type1": "b", "b": "b"}
    assert b.dict() == payload
    assert dict(b) == payload


def test_deserialize_poly_model():
    class Parent(PolyModel):
        class Config:
            type_root = True
            type_field = "type1"

    class ChildA(Parent):
        type1 = "a"
        a: int

    class ChildB(Parent):
        type1 = "b"
        b: str

    a = parse_obj_as(Parent, {"type1": "a", "a": 1})
    assert isinstance(a, ChildA)
    assert a.a == 1

    b = parse_obj_as(Parent, {"type1": "b", "b": "b"})
    assert isinstance(b, ChildB)
    assert b.b == "b"


def test_nested_poly():
    class InnerParent(PolyModel):
        class Config:
            type_root = True

        field: str

    class InnerChild(InnerParent):
        type: ClassVar[str] = "ic"

    class OuterParent(PolyModel):
        class Config:
            type_root = True

        inner: InnerParent

    class OuterChild(OuterParent):
        type: ClassVar[str] = "oc"

    obj = OuterChild(inner=InnerChild(field="a"))
    payload = {"type": "oc", "inner": {"type": "ic", "field": "a"}}
    assert obj.dict() == payload
    obj2 = parse_obj_as(OuterParent, payload)
    assert isinstance(obj2, OuterChild)
    assert isinstance(obj2.inner, InnerChild)
    assert obj2 == obj


def test_transient():
    class TransTest(PolyModel):
        class Config:
            type_root = True
            exclude = {"t_field"}

        field: str
        t_field: Optional[str] = None

    obj = TransTest(field="a", t_field="b")
    payload = {"field": "a", "type": "tests.polydantic.test_serde.TransTest"}
    assert obj.dict() == payload

    obj2 = parse_obj_as(TransTest, payload)
    assert obj2 == TransTest(field="a", t_field=None)


def test_multi_parent():
    class Parent1(BaseModel):
        field1: str

    class Parent2(PolyModel):
        class Config:
            type_root = True

        field2: str

    class Child(Parent1, Parent2):
        type: ClassVar[str] = "child"

    obj = Child(field2="2", field1="1")
    assert obj.field1 == "1"
    assert obj.field2 == "2"
    payload = {"type": "child", "field1": "1", "field2": "2"}
    assert obj.dict() == payload
    obj2 = parse_obj_as(Parent2, payload)
    assert isinstance(obj2, Child)
    assert obj2 == obj


class PayloadParent(PolyModel):
    class Config:
        exclude = {"trans"}

    trans: Any = None


class Payload(PayloadParent):
    value: int

    @validator("value")
    def counter(cls, value):  # pylint: disable=no-self-argument  # noqa: B902
        return value + 1


class Parent(PolyModel):
    class Config:
        type_root = True


class Model(Parent):
    field_cache: Any
    field: Payload
    field, field_raw, field_cache = lazy_field(Payload, "field", "field_cache")


def test_lazy_transient():
    payload = Payload(value=1)
    payload.trans = "value"
    model = Model(field=payload)
    assert isinstance(model.__dict__["field_cache"], Payload)
    model_dict = model.dict()
    assert "trans" not in model_dict["field"]
    new_obj = parse_obj_as(Parent, model_dict)
    assert isinstance(new_obj.__dict__["field_cache"], dict)
    assert "trans" not in new_obj.dict()["field"]
