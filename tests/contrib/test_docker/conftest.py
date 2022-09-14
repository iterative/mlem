import os
import time

import docker.errors
import pytest
from testcontainers.core.container import DockerContainer as TestContainer

from mlem.contrib.docker.base import DockerDaemon, DockerEnv, RemoteRegistry
from mlem.contrib.docker.context import use_mlem_source
from mlem.contrib.docker.utils import is_docker_running
from tests.conftest import long

EXTERNAL_REGISTRY_PORT = 2374
INTERNAL_REGISTRY_PORT = 5000
DAEMON_PORT = 2375
CLEAN = True
IMAGE_NAME = "mlem_test_docker_builder_image"


@pytest.fixture(scope="session")
def dockerenv_local(tmp_path_factory):
    return DockerEnv().dump(str(tmp_path_factory.mktemp("dockerenv_local")))


@pytest.fixture(scope="session")
def dind():
    with (
        TestContainer("docker:dind")
        .with_env("DOCKER_TLS_CERTDIR", "")
        .with_kargs(privileged=True)
        .with_exposed_ports(DAEMON_PORT)
        .with_bind_ports(EXTERNAL_REGISTRY_PORT, EXTERNAL_REGISTRY_PORT)
    ) as daemon:
        time.sleep(1)
        yield daemon


@pytest.fixture(scope="session")
def docker_daemon(dind):
    daemon = DockerDaemon(
        host=f"tcp://localhost:{dind.get_exposed_port(DAEMON_PORT)}"
    )
    exc = None
    for _ in range(10):
        try:
            with daemon.client() as c:
                c.info()
            return daemon
        except docker.errors.DockerException as e:
            exc = e
            time.sleep(2)
    if exc:
        raise exc
    return None


@pytest.fixture(scope="session")
def docker_registry(dind, docker_daemon):
    with docker_daemon.client() as c:
        c: docker.DockerClient
        c.containers.run(
            "registry:latest",
            ports={INTERNAL_REGISTRY_PORT: EXTERNAL_REGISTRY_PORT},
            detach=True,
            remove=True,
            environment={"REGISTRY_STORAGE_DELETE_ENABLED": "true"},
        )
        yield RemoteRegistry(host=f"localhost:{EXTERNAL_REGISTRY_PORT}")


@pytest.fixture(scope="session")
def dockerenv_remote(docker_registry, docker_daemon, tmp_path_factory):
    return DockerEnv(registry=docker_registry, daemon=docker_daemon).dump(
        str(tmp_path_factory.mktemp("dockerenv_remote"))
    )


def has_docker():
    if os.environ.get("SKIP_DOCKER_TESTS", None) == "true":
        return False
    current_os = os.environ.get("GITHUB_MATRIX_OS")
    current_python = os.environ.get("GITHUB_MATRIX_PYTHON")
    if (
        current_os is not None
        and current_os != "ubuntu-latest"
        or current_python is not None
        and current_python != "3.8"
    ):
        return False
    return is_docker_running()


def docker_test(f):
    mark = pytest.mark.docker
    skip = pytest.mark.skipif(
        not has_docker(), reason="docker is unavailable or skipped"
    )
    return long(mark(skip(f)))


@pytest.fixture(scope="session", autouse=True)
def mlem_source():
    with use_mlem_source("whl"):
        yield
