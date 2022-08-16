import pytest
from pydantic import BaseModel

from mlem.cli.types import iterate_type_fields
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
        for subtype in list_implementations(root_type)
    ],
)
def test_types_abs_name_subtype(runner: Runner, abs_name, subtype):
    result = runner.invoke(f"types {abs_name} {subtype}")
    assert result.exit_code == 0, result.exception
    # TODO assert "Field docstring missing" not in result.output


def test_iter_type_fields_subclass():
    class Parent(BaseModel):
        parent: str

    class Child(Parent):
        child: str

    fields = list(iterate_type_fields(Child))

    assert len(fields) == 2
