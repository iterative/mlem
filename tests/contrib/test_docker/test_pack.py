import os
import time

import pytest
import requests
from pytest_lazyfixture import lazy_fixture
from testcontainers.general import TestContainer

from mlem.api import build
from mlem.contrib.docker import DockerDirBuilder, DockerImageBuilder
from mlem.contrib.docker.base import DockerImage
from mlem.contrib.docker.context import DockerModelDirectory
from mlem.contrib.fastapi import FastAPIServer
from tests.conftest import long
from tests.contrib.test_docker.conftest import docker_test

SERVER_PORT = 8080


@long
@pytest.mark.parametrize(
    "modelmeta", [lazy_fixture("model_meta"), lazy_fixture("model_meta_saved")]
)
def test_build_dir(tmpdir, modelmeta):
    built = build(
        DockerDirBuilder(server=FastAPIServer(), target=str(tmpdir)),
        modelmeta,
    )
    assert isinstance(built, DockerModelDirectory)
    assert os.path.isfile(tmpdir / "run.sh")
    assert os.path.isfile(tmpdir / "Dockerfile")
    assert os.path.isfile(tmpdir / "requirements.txt")
    assert os.path.isfile(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model.mlem")


@docker_test
def test_pack_image(
    model_meta_saved_single, dockerenv_local, uses_docker_build
):
    built = build(
        DockerImageBuilder(
            server=FastAPIServer(),
            image=DockerImage(name="pack_docker_test_image"),
            force_overwrite=True,
        ),
        model_meta_saved_single,
    )
    assert isinstance(built, DockerImage)
    assert dockerenv_local.image_exists(built)
    with (
        TestContainer(built.name)
        .with_env("DOCKER_TLS_CERTDIR", "")
        .with_exposed_ports(SERVER_PORT)
    ) as service:
        time.sleep(10)
        r = requests.post(
            f"http://localhost:{service.get_exposed_port(SERVER_PORT)}/predict",
            json={"data": [[0, 0, 0, 0]]},
        )
        assert r.status_code == 200
