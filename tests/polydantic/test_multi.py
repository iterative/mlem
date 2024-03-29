from typing import ClassVar

import pytest
from pydantic import Extra, parse_obj_as

from mlem.polydantic import PolyModel


class CommonParent(PolyModel):
    root_type: ClassVar = ...

    class Config:
        type_root = True
        type_field = "root_type"


class Child(CommonParent):
    root_type = "child"


class InnerParent(CommonParent):
    root_type = "inner_parent"
    inner_type: ClassVar = "inner_parent2"

    class Config:
        type_root = True
        type_field = "inner_type"


class SubChild1(InnerParent):
    inner_type = "child1"


class SubChild2(InnerParent):
    inner_type = "child2"


def test_declaration():
    assert CommonParent.__is_root__
    assert not Child.__is_root__
    assert InnerParent.__is_root__
    assert not SubChild1.__is_root__
    assert not SubChild2.__is_root__

    assert CommonParent.__parent__ is CommonParent
    assert Child.__parent__ is CommonParent
    assert InnerParent.__parent__ is CommonParent
    assert SubChild1.__parent__ is InnerParent
    assert SubChild2.__parent__ is InnerParent

    assert CommonParent.__type_map__ == {
        "child": Child,
        "inner_parent": InnerParent,
    }
    assert Child.__type_map__ is CommonParent.__type_map__
    assert InnerParent.__type_map__ == {
        "child1": SubChild1,
        "child2": SubChild2,
        "inner_parent2": InnerParent,
    }
    assert SubChild1.__type_map__ is InnerParent.__type_map__
    assert SubChild2.__type_map__ is InnerParent.__type_map__

    assert list(CommonParent.__iter_parents__()) == [CommonParent]
    assert list(Child.__iter_parents__()) == [Child, CommonParent]
    assert list(InnerParent.__iter_parents__()) == [InnerParent, CommonParent]
    assert list(SubChild1.__iter_parents__()) == [
        SubChild1,
        InnerParent,
        CommonParent,
    ]
    assert list(SubChild2.__iter_parents__()) == [
        SubChild2,
        InnerParent,
        CommonParent,
    ]

    assert not list(CommonParent.__iter_parents__(include_top=False))
    assert list(Child.__iter_parents__(include_top=False)) == [
        Child,
    ]
    assert list(InnerParent.__iter_parents__(include_top=False)) == [
        InnerParent,
    ]
    assert list(SubChild1.__iter_parents__(include_top=False)) == [
        SubChild1,
        InnerParent,
    ]
    assert list(SubChild2.__iter_parents__(include_top=False)) == [
        SubChild2,
        InnerParent,
    ]


def test_serde_child():
    obj = Child()
    payload = obj.dict()
    assert payload == {"root_type": "child"}
    assert parse_obj_as(CommonParent, payload) == obj


def test_serde_inner_parent():
    obj = InnerParent()
    payload = obj.dict()
    assert payload == {
        "root_type": "inner_parent",
        "inner_type": "inner_parent2",
    }
    assert parse_obj_as(CommonParent, payload) == obj
    assert parse_obj_as(InnerParent, payload) == obj


@pytest.mark.parametrize("cls", [SubChild1, SubChild2])
def test_serde_subchilds(cls):
    obj = cls()
    payload = obj.dict()
    assert payload == {
        "root_type": "inner_parent",
        "inner_type": cls.inner_type,
    }
    assert parse_obj_as(CommonParent, payload) == obj
    assert parse_obj_as(InnerParent, payload) == obj


def test_serde_no_extra():
    class NEParent(PolyModel):
        class Config:
            extra = Extra.forbid
            type_root = True

        p: str

    class NEChild(NEParent):
        type: ClassVar = "child"

        c: str

    c = NEChild(p="a", c="b")
    payload = c.dict()
    assert payload == {"p": "a", "c": "b", "type": "child"}
    c2 = parse_obj_as(NEParent, payload)
    assert isinstance(c2, NEChild)
    assert c2 == c


def test_serde_no_extra_nested():
    class NEGrandParent(PolyModel):
        class Config:
            extra = Extra.forbid
            type_root = True

        g: str

    class NEParent(NEGrandParent):
        type: ClassVar = "parent"

        class Config:
            type_root = True
            type_field = "object_type"

        p: str

    class NEChild(NEParent):
        object_type: ClassVar = "child"

        c: str

    c = NEChild(p="a", c="b", g="0")
    payload = c.dict()
    assert payload == {
        "p": "a",
        "c": "b",
        "type": "parent",
        "object_type": "child",
        "g": "0",
    }

    c2 = parse_obj_as(NEGrandParent, payload)
    assert isinstance(c2, NEChild)
    assert c2 == c

    c3 = parse_obj_as(NEParent, payload)
    assert isinstance(c3, NEChild)
    assert c3 == c

    c4 = parse_obj_as(NEChild, payload)
    assert isinstance(c4, NEChild)
    assert c4 == c
