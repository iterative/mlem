from typing import Optional

import pytest
from pydantic import BaseModel

from mlem.cli.types import iterate_type_fields
from mlem.cli.utils import get_field_help
from mlem.core.base import MlemABC
from mlem.utils.entrypoints import list_implementations
from tests.cli.conftest import Runner


def test_types(runner: Runner):
    result = runner.invoke("types")
    assert result.exit_code == 0, (result.exception, result.output)
    assert all(typename in result.output for typename in MlemABC.abs_types)


@pytest.mark.parametrize("abs_name", MlemABC.abs_types.keys())
def test_types_abs_name(runner: Runner, abs_name):
    result = runner.invoke(f"types {abs_name}")
    assert result.exit_code == 0, result.exception
    assert set(result.output.splitlines()) == set(
        list_implementations(abs_name, include_hidden=False)
    )


@pytest.mark.parametrize(
    "abs_name,subtype",
    [
        (abs_name, subtype)
        for abs_name, root_type in MlemABC.abs_types.items()
        for subtype in list_implementations(root_type, include_hidden=False)
    ],
)
def test_types_abs_name_subtype(runner: Runner, abs_name, subtype):
    result = runner.invoke(f"types {abs_name} {subtype}")
    assert result.exit_code == 0, result.exception
    assert "docstring missing" not in result.output


def test_iter_type_fields_subclass():
    class Parent(BaseModel):
        parent: str
        """parent"""

    class Child(Parent):
        child: str
        """child"""
        excluded: Optional[str] = None

        class Config:
            fields = {"excluded": {"exclude": True}}

    fields = list(iterate_type_fields(Child))

    assert len(fields) == 2
    assert {get_field_help(Child, f.path) for f in fields} == {
        "parent",
        "child",
    }


def test_iter_type_fields_subclass_multiinheritance():
    class Parent(BaseModel):
        parent: str
        """parent"""

    class Parent2(BaseModel):
        parent2: str
        """parent2"""

    class Child(Parent, Parent2):
        child: str
        """child"""

    fields = list(iterate_type_fields(Child))

    assert len(fields) == 3
    assert {get_field_help(Child, f.path) for f in fields} == {
        "parent",
        "child",
        "parent2",
    }
