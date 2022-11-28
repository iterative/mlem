from typing import Any, ClassVar

import pytest

import mlem
from mlem.core.data_type import DataType, DataWriter
from mlem.core.requirements import Requirements
from mlem.runtime import Interface
from mlem.runtime.interface import (
    InterfaceArgument,
    InterfaceDataType,
    InterfaceMethod,
    SimpleInterface,
    expose,
)


class Container(DataType):
    type: ClassVar[str] = "test_container"
    field: int

    def serialize(
        self, instance: Any  # pylint: disable=unused-argument
    ) -> dict:
        return {}

    def deserialize(self, obj: dict) -> Any:
        pass

    def get_requirements(self) -> Requirements:
        return Requirements.new([])

    def get_writer(
        self, project: str = None, filename: str = None, **kwargs
    ) -> DataWriter:
        raise NotImplementedError


@pytest.fixture
def interface() -> Interface:
    class MyInterface(SimpleInterface):
        @expose
        def method1(self, arg1: Container(field=5)) -> Container(field=5):  # type: ignore[valid-type]
            self.method2()
            return arg1

        def method2(self):
            pass

    return MyInterface()


def test_interface_descriptor__from_interface(interface: Interface):
    d = interface.get_descriptor()
    sig = InterfaceMethod(
        name="method1",
        args=[
            InterfaceArgument(
                name="arg1",
                data_type=Container(field=5),
            )
        ],
        returns=InterfaceDataType(data_type=Container(field=5)),
    )
    assert d.__root__ == {"method1": sig}


def test_interface_descriptor__to_dict(interface: Interface):
    d = interface.get_versioned_descriptor()

    assert d.dict() == {
        "version": mlem.__version__,
        "methods": {
            "method1": {
                "args": [
                    {
                        "default": None,
                        "name": "arg1",
                        "required": True,
                        "data_type": {"field": 5, "type": "test_container"},
                        "serializer": None,
                    }
                ],
                "name": "method1",
                "returns": {
                    "data_type": {"field": 5, "type": "test_container"},
                    "serializer": None,
                },
            }
        },
    }
