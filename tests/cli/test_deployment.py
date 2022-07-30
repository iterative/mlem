import os
from typing import ClassVar

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
)
from mlem.runtime.client import Client, HTTPClient
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


@pytest.fixture
def mock_deploy_get_client(mocker, request_get_mock, request_post_mock):
    return mocker.patch(
        "tests.cli.test_deployment.DeployStateMock.get_client",
        return_value=HTTPClient(host="", port=None),
    )


class DeployStateMock(DeployState):
    allow_default: ClassVar = True

    def get_client(self) -> Client:
        pass


class MlemDeploymentMock(MlemDeployment):
    class Config:
        use_enum_values = True

    type: ClassVar = "mock"
    state_type: ClassVar = DeployStateMock

    status: DeployStatus = DeployStatus.NOT_DEPLOYED
    param: str = ""


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
        model=model_meta_saved_single.make_link(),
        model_cache=model_meta_saved_single,
        env=mock_env_path,
    ).dump(path)
    return path


def test_deploy_meta_str_model(mlem_project, model_meta, mock_env_path):
    model_meta.dump("model", project=mlem_project)

    deployment = MlemDeploymentMock(model="model", env=mock_env_path)
    deployment.dump("deployment", project=mlem_project)

    with deployment.loc.open("r") as f:
        data = safe_load(f)
        assert data == {
            "model": "model",
            "object_type": "deployment",
            "type": "mock",
            "env": make_posix(mock_env_path),
        }

    deployment2 = load_meta(
        "deployment", project=mlem_project, force_type=MlemDeployment
    )
    assert deployment2 == deployment
    assert deployment2.get_model() == model_meta
    assert deployment2.get_env() == load_meta(mock_env_path)


def test_deploy_meta_link_str_model(mlem_project, model_meta, mock_env_path):
    model_meta.dump("model", project=mlem_project)

    deployment = MlemDeploymentMock(
        model=MlemLink(path="model", link_type="model"),
        env=MlemLink(path=mock_env_path, link_type="env"),
    )
    deployment.dump("deployment", project=mlem_project)

    with deployment.loc.open("r") as f:
        data = safe_load(f)
        assert data == {
            "model": "model",
            "object_type": "deployment",
            "type": "mock",
            "env": make_posix(mock_env_path),
        }

    deployment2 = load_meta(
        "deployment", project=mlem_project, force_type=MlemDeployment
    )
    assert deployment2 == deployment
    assert deployment2.get_model() == model_meta
    assert deployment2.get_env() == load_meta(mock_env_path)


def test_deploy_meta_link_model(mlem_project, model_meta, mock_env_path):
    model_meta.dump("model", project=mlem_project)

    deployment = MlemDeploymentMock(
        model=MlemLink(path="model", project=mlem_project, link_type="model"),
        env=MlemLink(
            path=mock_env_path, project=mlem_project, link_type="env"
        ),
    )
    deployment.dump("deployment", project=mlem_project)

    with deployment.loc.open("r") as f:
        data = safe_load(f)
        assert data == {
            "model": {"path": "model", "project": mlem_project},
            "object_type": "deployment",
            "type": "mock",
            "env": {
                "path": make_posix(mock_env_path),
                "project": mlem_project,
            },
        }

    deployment2 = load_meta(
        "deployment", project=mlem_project, force_type=MlemDeployment
    )
    assert deployment2 == deployment
    assert deployment2.get_model() == model_meta
    assert deployment2.get_env() == load_meta(mock_env_path)


def test_deploy_meta_link_model_no_project(tmpdir, model_meta, mock_env_path):
    model_path = os.path.join(tmpdir, "model")
    model_meta.dump(model_path)

    deployment = MlemDeploymentMock(
        model=MlemLink(path="model", link_type="model"),
        env=MlemLink(path=mock_env_path, link_type="env"),
    )
    deployment_path = os.path.join(tmpdir, "deployment")
    deployment.dump(deployment_path)

    with deployment.loc.open("r") as f:
        data = safe_load(f)
        assert data == {
            "model": "model",
            "object_type": "deployment",
            "type": "mock",
            "env": make_posix(mock_env_path),
        }

    deployment2 = load_meta(deployment_path, force_type=MlemDeployment)
    assert deployment2 == deployment
    assert deployment2.get_model() == model_meta
    assert deployment2.get_env() == load_meta(mock_env_path)


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
    assert result.exit_code == 0, (result.output, result.exception)
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.status == DeployStatus.NOT_DEPLOYED
    predictions = load(path)
    assert isinstance(predictions, ndarray)
