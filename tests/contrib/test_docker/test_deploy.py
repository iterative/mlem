# pylint: disable=cannot-enumerate-pytest-fixtures
import os
import tempfile
import time

import pytest
from requests.exceptions import HTTPError

from mlem.contrib.docker.base import (
    DockerContainer,
    DockerContainerState,
    DockerEnv,
    DockerImage,
)
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.errors import DeploymentError
from mlem.core.objects import DeployStatus
from tests.contrib.test_docker.conftest import docker_test

IMAGE_NAME = "mike0sv/ebaklya"
BROKEN_IMAGE_NAME = "test-broken-image"
CONTAINER_NAME = "mlem-runner-test-docker-container"
REPOSITORY_NAME = "mlem"

REGISTRY_PORT = 5000


@pytest.fixture(scope="session")
def _test_images(tmpdir_factory, dockerenv_local, dockerenv_remote):
    with dockerenv_local.daemon.client() as client:
        tag_name = f"{dockerenv_remote.registry.get_host()}/{REPOSITORY_NAME}/{IMAGE_NAME}"
        client.images.pull(IMAGE_NAME, "latest").tag(tag_name)
        client.images.push(tag_name)

        tmpdir = str(tmpdir_factory.mktemp("image"))
        # create failing image: alpine is too small to have python inside
        with open(
            os.path.join(tmpdir, "Dockerfile"), "w", encoding="utf8"
        ) as f:
            f.write(
                """
                FROM alpine:latest
                CMD python
           """
            )
        broken_tag_name = f"{dockerenv_remote.registry.get_host()}/{REPOSITORY_NAME}/{BROKEN_IMAGE_NAME}"
        client.images.build(path=tmpdir, tag=broken_tag_name)
        client.images.push(broken_tag_name)


@docker_test
def test_run_default_registry(
    dockerenv_local, _test_images, model_meta_saved_single
):
    _check_runner(IMAGE_NAME, dockerenv_local, model_meta_saved_single)


@docker_test
def test_run_remote_registry(
    dockerenv_remote, _test_images, model_meta_saved_single
):
    _check_runner(IMAGE_NAME, dockerenv_remote, model_meta_saved_single)


@docker_test
def test_run_local_image_name_that_will_never_exist(
    dockerenv_local, model_meta_saved_single
):
    with pytest.raises(HTTPError):
        _check_runner(
            "mlem_image_name_that_will_never_exist",
            dockerenv_local,
            model_meta_saved_single,
        )


@docker_test
def test_run_local_fail_inside_container(
    dockerenv_remote, _test_images, model_meta_saved_single
):
    with pytest.raises(DeploymentError):
        _check_runner(
            f"{dockerenv_remote.registry.get_host()}/{REPOSITORY_NAME}/{BROKEN_IMAGE_NAME}",
            dockerenv_remote,
            model_meta_saved_single,
        )


def _check_runner(img, env: DockerEnv, model):
    with tempfile.TemporaryDirectory() as tmpdir:
        instance = DockerContainer(
            container_name=CONTAINER_NAME,
            port_mapping={80: 8008},
            state=DockerContainerState(image=DockerImage(name=img)),
            server=FastAPIServer(),
            model_link=model.make_link(),
            env_link=env.make_link(),
            rm=False,
        )
        instance.update_model_hash(model)
        instance.dump(os.path.join(tmpdir, "deploy"))
        assert env.get_status(instance) == DeployStatus.NOT_DEPLOYED

        env.deploy(instance)

        instance.wait_for_status(
            DeployStatus.RUNNING, allowed_intermediate=[DeployStatus.STARTING]
        )
        time.sleep(0.1)

        assert env.get_status(instance) == DeployStatus.RUNNING

        env.remove(instance)
        time.sleep(0.1)

        assert env.get_status(instance) == DeployStatus.NOT_DEPLOYED
