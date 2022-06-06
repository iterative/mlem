import os
from typing import ClassVar

import pytest
from numpy import ndarray

from mlem.api import load
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemLink,
)
from mlem.runtime.client import Client, HTTPClient
from tests.cli.conftest import Runner


@pytest.fixture
def mock_deploy_get_client(mocker, request_get_mock, request_post_mock):
    return mocker.patch(
        "tests.cli.test_deployment.DeployStateMock.get_client",
        return_value=HTTPClient(host="", port=None),
    )


class DeployStateMock(DeployState):
    def get_client(self) -> Client:
        pass


class MlemDeploymentMock(MlemDeployment):
    class Config:
        use_enum_values = True

    type: ClassVar = "mock"
    status: DeployStatus = DeployStatus.NOT_DEPLOYED
    param: str = ""
    state: DeployState = DeployStateMock()


class MlemEnvMock(MlemEnv):
    type: ClassVar = "mock"
    deploy_type: ClassVar = MlemDeploymentMock

    def deploy(self, meta: MlemDeploymentMock):
        meta.status = DeployStatus.RUNNING
        meta.update()

    def remove(self, meta: MlemDeploymentMock):
        meta.status = DeployStatus.STOPPED
        meta.update()

    def get_status(
        self, meta: MlemDeploymentMock, raise_on_error=True
    ) -> "DeployStatus":
        return meta.status


@pytest.fixture
def mock_env_path(tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "mock-target-env")
    MlemEnvMock().dump(path)
    return path


@pytest.fixture()
def mock_deploy_path(tmp_path, mock_env_path, model_meta_saved_single):
    path = os.path.join(tmp_path, "deployname")
    MlemDeploymentMock(
        param="bbb",
        model_link=model_meta_saved_single.make_link(),
        env_link=MlemLink(path=mock_env_path, link_type="env"),
    ).dump(path)
    return path


def test_deploy_create_new(
    runner: Runner, model_meta_saved_single, mock_env_path, tmp_path
):
    path = os.path.join(tmp_path, "deployname")
    result = runner.invoke(
        f"deploy run {path} -m {model_meta_saved_single.loc.uri} -t {mock_env_path} -c param=aaa".split()
    )
    assert result.exit_code == 0, result.output
    assert os.path.isfile(path + MLEM_EXT)
    meta = load_meta(path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.param == "aaa"
    assert meta.status == DeployStatus.RUNNING


def test_deploy_create_existing(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy run {mock_deploy_path}".split())
    assert result.exit_code == 0, result.output
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.param == "bbb"
    assert meta.status == DeployStatus.RUNNING


def test_deploy_status(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy status {mock_deploy_path}".split())
    assert result.exit_code == 0, result.output
    assert result.output.strip() == DeployStatus.NOT_DEPLOYED.value


def test_deploy_remove(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy remove {mock_deploy_path}".split())
    assert result.exit_code == 0, result.output
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.status == DeployStatus.STOPPED


def test_deploy_apply(
    runner: Runner,
    mock_deploy_path,
    data_path,
    mock_deploy_get_client,
    tmp_path,
):
    path = os.path.join(tmp_path, "output")
    result = runner.invoke(
        f"deploy apply {mock_deploy_path} {data_path} -o {path}".split()
    )
    assert result.exit_code == 0, result.output
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.status == DeployStatus.NOT_DEPLOYED
    predictions = load(path)
    assert isinstance(predictions, ndarray)
