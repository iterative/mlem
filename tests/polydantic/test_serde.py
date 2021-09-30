from typing import ClassVar, Optional

from pydantic import BaseModel, parse_obj_as

from mlem.polydantic import PolyModel


def test_poly_model_dict():
    class Parent(PolyModel):
        __type_root__ = True
        __type_field__ = "type1"

    class ChildA(Parent):
        type1: ClassVar = "a"
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
        __type_root__ = True
        __type_field__ = "type1"

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
        __type_root__: ClassVar = True
        field: str

    class InnerChild(InnerParent):
        type: ClassVar = "ic"

    class OuterParent(PolyModel):
        __type_root__: ClassVar = True
        inner: InnerParent

    class OuterChild(OuterParent):
        type: ClassVar = "oc"

    obj = OuterChild(inner=InnerChild(field="a"))
    payload = {"type": "oc", "inner": {"type": "ic", "field": "a"}}
    assert obj.dict() == payload
    obj2 = parse_obj_as(OuterParent, payload)
    assert isinstance(obj2, OuterChild)
    assert isinstance(obj2.inner, InnerChild)
    assert obj2 == obj


def test_transient():
    class TransTest(PolyModel):
        __transient_fields__ = {"t_field"}
        __type_root__ = True
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
        __type_root__ = True
        field2: str

    class Child(Parent1, Parent2):
        type: ClassVar = "child"

    obj = Child(field2="2", field1="1")
    assert obj.field1 == "1"
    assert obj.field2 == "2"
    payload = {"type": "child", "field1": "1", "field2": "2"}
    assert obj.dict() == payload
    obj2 = parse_obj_as(Parent2, payload)
    assert isinstance(obj2, Child)
    assert obj2 == obj
