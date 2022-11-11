from typing import Any, ClassVar

import pytest

import mlem
from mlem.core.data_type import DataType, DataWriter
from mlem.core.model import Argument, Signature
from mlem.core.requirements import Requirements
from mlem.runtime import Interface
from mlem.runtime.interface import SimpleInterface, expose


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
    sig = Signature(
        name="method1",
        args=[
            Argument(
                name="arg1",
                type_=Container(field=5),
            )
        ],
        returns=Container(field=5),
    )
    assert d.methods == {"method1": sig}


def test_interface_descriptor__to_dict(interface: Interface):
    d = interface.get_descriptor()

    assert d.dict() == {
        "version": mlem.__version__,
        "methods": {
            "method1": {
                "args": [
                    {
                        "default": None,
                        "kw_only": False,
                        "name": "arg1",
                        "required": True,
                        "type_": {"field": 5, "type": "test_container"},
                    }
                ],
                "name": "method1",
                "returns": {"field": 5, "type": "test_container"},
                "varargs": None,
                "varkw": None,
            }
        },
    }
