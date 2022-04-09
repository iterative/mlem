from typing import ClassVar, Optional

from mlem.contrib.docker import DockerImagePackager
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.base import (
    MlemObject,
    build_mlem_object,
    parse_links,
    smart_split,
)
from mlem.core.objects import MlemLink, MlemMeta, ModelMeta
from mlem.pack import Packager
from mlem.runtime.server.base import Server
from tests.conftest import resource_path


def test_build_model():
    res = build_mlem_object(
        Packager,
        "docker",
        ["image.name=kek"],
        [f"server={resource_path(__file__, 'server.yaml')}"],
    )
    assert isinstance(res, DockerImagePackager)
    assert res.image.name == "kek"
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081


def test_smart_split():
    assert smart_split("a 'b c' d", " ") == ["a", "b c", "d"]
    assert smart_split('a."b.c".d', ".") == ["a", "b.c", "d"]
    assert smart_split("a.'b c'.d", ".") == ["a", "b c", "d"]


def test_parse_links():
    class ModelWithLink(MlemMeta):
        field_link: MlemLink
        field: Optional[ModelMeta]

    assert parse_links(ModelWithLink, ["field=somepath"]) == (
        [],
        {"field_link": MlemLink(path="somepath", link_type="model")},
    )


class MockMlemObject(MlemObject):
    abs_name: ClassVar = "mock"
    server: Server


def test_build_with_replace():
    res = build_mlem_object(
        MockMlemObject,
        "mock",
        ["server=fastapi", "server.port=8081"],
    )
    assert isinstance(res, MockMlemObject)
    assert isinstance(res.server, FastAPIServer)
    assert res.server.port == 8081

    res = build_mlem_object(
        MockMlemObject,
        "mock",
        ["server=fastapi"],
    )
    assert isinstance(res, MockMlemObject)
    assert isinstance(res.server, FastAPIServer)
