from abc import abstractmethod

from mlem.core.base import MlemABC
from mlem.core.objects import MlemEnv, MlemObject
from mlem.utils.entrypoints import list_implementations


class MockABC(MlemABC):
    abs_name = "mock"

    class Config:
        type_root = True

    @abstractmethod
    def something(self):
        pass


class MockImpl(MockABC):
    type = "impl"

    def something(self):
        pass


def test_list_implementations():
    assert list_implementations(MockABC) == ["impl"]
    assert list_implementations("mock") == ["impl"]


def test_list_implementations_meta():
    assert "model" in list_implementations("meta")
    assert "model" in list_implementations(MlemObject)

    assert "docker" in list_implementations("meta", MlemEnv)
    assert "docker" in list_implementations(MlemObject, MlemEnv)

    assert "docker" in list_implementations("meta", "env")
    assert "docker" in list_implementations(MlemObject, "env")
