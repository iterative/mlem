# pylint: disable=cannot-enumerate-pytest-fixtures
import os
import tempfile
import time

import pytest
from requests.exceptions import HTTPError

from mlem.api import deploy
from mlem.contrib.docker.base import (
    DockerContainer,
    DockerContainerState,
    DockerEnv,
    DockerImage,
)
from mlem.contrib.docker.context import DockerBuildArgs
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.errors import DeploymentError
from mlem.core.objects import DeployStatus
from tests.conftest import resource_path
from tests.contrib.test_docker.conftest import docker_test

IMAGE_NAME = "mike0sv/ebaklya"
BROKEN_IMAGE_NAME = "test-broken-image"
CONTAINER_NAME = "mlem-runner-test-docker-container"
REPOSITORY_NAME = "mlem"

REGISTRY_PORT = 5000


@pytest.fixture(scope="session")
def _test_images(dockerenv_local):
    with dockerenv_local.daemon.client() as client:
        client.images.pull(IMAGE_NAME, "latest")


@pytest.fixture(scope="session")
def _test_images_remote(
    tmpdir_factory, dockerenv_local, dockerenv_remote, _test_images
):
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
    dockerenv_remote, _test_images_remote, model_meta_saved_single
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
    dockerenv_remote, _test_images_remote, model_meta_saved_single
):
    with pytest.raises(DeploymentError):
        _check_runner(
            f"{dockerenv_remote.registry.get_host()}/{REPOSITORY_NAME}/{BROKEN_IMAGE_NAME}",
            dockerenv_remote,
            model_meta_saved_single,
        )


@docker_test
def test_deploy_full(
    tmp_path_factory, dockerenv_local, model_meta_saved_single
):
    meta_path = tmp_path_factory.mktemp("deploy-meta")
    meta = deploy(
        str(meta_path),
        model_meta_saved_single,
        dockerenv_local,
        args=DockerBuildArgs(templates_dir=[resource_path(__file__)]),
        server="fastapi",
        container_name="test_full_deploy",
    )

    meta.wait_for_status(
        DeployStatus.RUNNING,
        allowed_intermediate=[
            DeployStatus.NOT_DEPLOYED,
            DeployStatus.STARTING,
        ],
        times=50,
    )
    assert meta.get_status() == DeployStatus.RUNNING


def _check_runner(img, env: DockerEnv, model):
    with tempfile.TemporaryDirectory() as tmpdir:
        instance = DockerContainer(
            container_name=CONTAINER_NAME,
            ports=["8008:80"],
            server=FastAPIServer(),
            env=env,
            rm=False,
        )
        instance.dump(os.path.join(tmpdir, "deploy"))
        instance.update_state(
            DockerContainerState(
                image=DockerImage(name=img),
                model_hash=model.meta_hash(),
                declaration=instance,
            )
        )
        assert instance.get_status() == DeployStatus.NOT_DEPLOYED

        instance.deploy(model)

        instance.wait_for_status(
            DeployStatus.RUNNING, allowed_intermediate=[DeployStatus.STARTING]
        )
        time.sleep(0.1)

        assert instance.get_status() == DeployStatus.RUNNING

        instance.remove()
        time.sleep(0.1)

        assert instance.get_status() == DeployStatus.NOT_DEPLOYED
