import os
from typing import Any, ClassVar, Optional

import pytest
from numpy import ndarray
from yaml import safe_load

from mlem.api import load
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemLink,
    MlemModel,
)
from mlem.runtime.client import Client, HTTPClient
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


class DeployStateMock(DeployState):
    """mock"""

    allow_default: ClassVar = True


class MlemDeploymentMock(MlemDeployment):
    """mock"""

    class Config:
        use_enum_values = True

    type: ClassVar = "mock"
    state_type: ClassVar = DeployStateMock

    status: DeployStatus = DeployStatus.NOT_DEPLOYED
    """status"""
    param: str = ""
    """param"""

    def _get_client(self, state) -> Client:
        return HTTPClient(host="", port=None)

    def deploy(self, model: MlemModel):
        self.status = DeployStatus.RUNNING
        self.update()

    def remove(self):
        self.status = DeployStatus.STOPPED
        self.update()

    def get_status(self, raise_on_error=True) -> "DeployStatus":
        return self.status


class MlemEnvMock(MlemEnv):
    """mock"""

    type: ClassVar = "mock"
    deploy_type: ClassVar = MlemDeploymentMock


@pytest.fixture
def mock_env_path(tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "mock-target-env")
    MlemEnvMock().dump(path)
    return path


@pytest.fixture()
def mock_deploy_path(tmp_path, mock_env_path):
    path = os.path.join(tmp_path, "deployname")
    MlemDeploymentMock(
        param="bbb",
        env=mock_env_path,
    ).dump(path)
    return path


def _check_deployment_meta(
    deployment: MlemDeployment,
    mlem_project: Optional[str],
    env_path: str,
    path: str = "deployment",
    env: Any = None,
):
    deployment.dump(path, project=mlem_project)

    with deployment.loc.open("r") as f:
        data = safe_load(f)
        assert data == {
            "object_type": "deployment",
            "type": "mock",
            "env": env or make_posix(env_path),
        }

    deployment2 = load_meta(
        path, project=mlem_project, force_type=MlemDeployment
    )
    assert deployment2 == deployment
    assert deployment2.get_env() == load_meta(env_path)


def test_deploy_meta_str_env(mlem_project, mock_env_path):
    deployment = MlemDeploymentMock(env=mock_env_path)
    _check_deployment_meta(deployment, mlem_project, mock_env_path)


def test_deploy_meta_link_env(mlem_project, mock_env_path):
    deployment = MlemDeploymentMock(
        env=MlemLink(path=mock_env_path, link_type="env"),
    )
    _check_deployment_meta(deployment, mlem_project, mock_env_path)


def test_deploy_meta_link_env_project(mlem_project, mock_env_path):
    load_meta(mock_env_path).clone("project_env", project=mlem_project)

    deployment = MlemDeploymentMock(
        env=MlemLink(
            path="project_env", project=mlem_project, link_type="env"
        ),
    )
    _check_deployment_meta(
        deployment,
        mlem_project,
        mock_env_path,
        env={
            "path": "project_env",
            "project": make_posix(mlem_project),
        },
    )


def test_deploy_meta_link_env_no_project(tmpdir, mock_env_path):

    deployment = MlemDeploymentMock(
        env=MlemLink(path=mock_env_path, link_type="env"),
    )
    deployment_path = os.path.join(tmpdir, "deployment")

    _check_deployment_meta(
        deployment, None, mock_env_path, path=deployment_path
    )


def test_read_relative_model_from_remote_deploy_meta():
    """TODO
    path = "s3://..."
    model.dump(path / "model");
    deployment = MlemDeploymentMock(
        model=model,
        env=MlemLink(
            path=mock_env_path, link_type="env"
        ),
    )
    deployment.dump(path / deployment)

    deployment2 = load_meta(...)
    deployment2.get_model()
    """


def test_deploy_create_new(
    runner: Runner, model_meta_saved_single, mock_env_path, tmp_path
):
    path = os.path.join(tmp_path, "deployname")
    result = runner.invoke(
        f"deploy run {path} -m {model_meta_saved_single.loc.uri} -t {mock_env_path} -c param=aaa".split()
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    assert os.path.isfile(path + MLEM_EXT)
    meta = load_meta(path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.param == "aaa"
    assert meta.status == DeployStatus.RUNNING


def test_deploy_create_existing(
    runner: Runner, mock_deploy_path, model_meta_saved_single
):
    result = runner.invoke(
        f"deploy run {mock_deploy_path} -m {model_meta_saved_single.loc.fullpath}".split(),
        raise_on_error=True,
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.param == "bbb"
    assert meta.status == DeployStatus.RUNNING


def test_deploy_status(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy status {mock_deploy_path}".split())
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    assert result.stdout.strip() == DeployStatus.NOT_DEPLOYED.value


def test_deploy_remove(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy remove {mock_deploy_path}".split())
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.status == DeployStatus.STOPPED


def test_deploy_apply(
    runner: Runner,
    mock_deploy_path,
    data_path,
    tmp_path,
    request_get_mock,
    request_post_mock,
):
    path = os.path.join(tmp_path, "output")
    result = runner.invoke(
        f"deploy apply {mock_deploy_path} {data_path} -o {path}".split()
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.status == DeployStatus.NOT_DEPLOYED
    predictions = load(path)
    assert isinstance(predictions, ndarray)
