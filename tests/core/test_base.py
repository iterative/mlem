from typing import ClassVar, List, Optional

from pydantic import BaseModel

from mlem.contrib.docker import DockerImageBuilder
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.base import MlemABC, build_mlem_object, parse_links, smart_split
from mlem.core.objects import MlemBuilder, MlemLink, MlemModel, MlemObject
from mlem.runtime.server import Server
from tests.conftest import resource_path


def test_build_model():
    res = build_mlem_object(
        MlemBuilder,
        "docker",
        ["image.name=kek"],
        [f"server={resource_path(__file__, 'server.yaml')}"],
    )
    assert isinstance(res, DockerImageBuilder)
    assert res.image.name == "kek"
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081


def test_smart_split():
    assert smart_split("a 'b c' d", " ") == ["a", "b c", "d"]
    assert smart_split('a."b.c".d', ".") == ["a", "b.c", "d"]
    assert smart_split("a.'b c'.d", ".") == ["a", "b c", "d"]


def test_smart_split_maxsplit():
    assert smart_split("a=b=c=d", "=", maxsplit=2) == ["a", "b", "c=d"]


def test_parse_links():
    class ModelWithLink(MlemObject):
        field_link: MlemLink
        field: Optional[MlemModel]

    assert parse_links(ModelWithLink, ["field=somepath"]) == (
        [],
        {"field_link": MlemLink(path="somepath", link_type="model")},
    )


class MockMlemABC(MlemABC):
    abs_name: ClassVar = "mock"
    server: Server


def test_build_with_replace():
    res = build_mlem_object(
        MockMlemABC,
        "mock",
        ["server=fastapi", "server.port=8081", "server.host=localhost"],
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081
    assert res.server.host == "localhost"

    res = build_mlem_object(
        MockMlemABC,
        "mock",
        ["server=fastapi"],
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)

    res = build_mlem_object(
        MockMlemABC,
        "mock",
        conf={
            "server": "fastapi",
            "server.port": 8081,
            "server.host": "localhost",
        },
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081
    assert res.server.host == "localhost"


def test_build_with_list():
    class MockMlemABCList(MlemABC):
        abs_name: ClassVar = "mock_list"
        values: List[str]

    res = build_mlem_object(
        MockMlemABCList,
        "mock_list",
        ["values.0=a", "values.1=b"],
    )
    assert isinstance(res, MockMlemABCList)
    assert isinstance(res.values, list)
    assert res.values == ["a", "b"]


def test_build_with_list_complex():
    class Value(BaseModel):
        field: str

    class MockMlemABCListComplex(MlemABC):
        abs_name: ClassVar = "mock_list_complex"
        values: List[Value]

    res = build_mlem_object(
        MockMlemABCListComplex,
        "mock_list_complex",
        ["values.0.field=a", "values.1.field=b"],
    )
    assert isinstance(res, MockMlemABCListComplex)
    assert isinstance(res.values, list)
    assert res.values == [Value(field="a"), Value(field="b")]
