import os
import time

import pytest
import requests
from testcontainers.general import TestContainer

from mlem.api import pack
from mlem.contrib.docker import DockerDirPackager, DockerImagePackager
from mlem.contrib.docker.base import DockerImage
from mlem.contrib.docker.context import DockerModelDirectory
from mlem.contrib.fastapi import FastAPIServer
from tests.contrib.test_docker.conftest import docker_test

SERVER_PORT = 8080


@pytest.mark.xfail(reason="fails on windows machines")
def test_pack_dir(tmpdir, model_meta_saved):
    packed = pack(
        DockerDirPackager(server=FastAPIServer()),
        model_meta_saved,
        str(tmpdir),
    )
    assert isinstance(packed, DockerModelDirectory)
    assert os.path.isfile(tmpdir / "run.sh")
    assert os.path.isfile(tmpdir / "Dockerfile")
    assert os.path.isfile(tmpdir / "requirements.txt")
    assert os.path.isfile(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model.mlem")


@docker_test
def test_pack_image(
    model_meta_saved_single, dockerenv_local, uses_docker_build
):
    packed = pack(
        DockerImagePackager(
            server=FastAPIServer(),
            image=DockerImage(name="pack_docker_test_image"),
            force_overwrite=True,
        ),
        model_meta_saved_single,
        "pack_docker_test",
    )
    assert isinstance(packed, DockerImage)
    assert dockerenv_local.image_exists(packed)
    with (
        TestContainer(packed.name)
        .with_env("DOCKER_TLS_CERTDIR", "")
        .with_exposed_ports(SERVER_PORT)
    ) as service:
        time.sleep(1)
        r = requests.post(
            f"http://localhost:{service.get_exposed_port(SERVER_PORT)}/predict",
            json={"data": [[0, 0, 0, 0]]},
        )
        assert r.status_code == 200
