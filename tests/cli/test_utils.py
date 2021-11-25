from mlem.cli.utils import build_poly_model, smart_split
from mlem.contrib.fastapi import FastAPIServer
from mlem.pack import Packager
from mlem.pack.docker import DockerImagePackager
from tests.conftest import resource_path


def test_build_model():
    res = build_poly_model(
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
