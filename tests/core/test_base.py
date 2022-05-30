from typing import ClassVar, Optional

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
        ["server=fastapi", "server.port=8081"],
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081

    res = build_mlem_object(
        MockMlemABC,
        "mock",
        ["server=fastapi"],
    )
    assert isinstance(res, MockMlemABC)
    assert isinstance(res.server, FastAPIServer)
