import os
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from mlem.contrib.fastapi import FastAPIServer
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    DeployMeta,
    DeployState,
    DeployStatus,
    MlemLink,
    ModelMeta,
    TargetEnvMeta,
)
from mlem.runtime.client.base import BaseClient, HTTPClient
from mlem.runtime.interface.base import ModelInterface
from tests.cli.conftest import Runner


@pytest.fixture
def interface(model, train):
    model = ModelMeta.from_obj(model, sample_data=train)
    interface = ModelInterface.from_model(model)
    return interface


@pytest.fixture
def client(interface):
    app = FastAPIServer().app_init(interface)
    return TestClient(app)


@pytest.fixture
def request_get_mock(mocker, client):
    def patched_get(url, params=None, **kwargs):
        url = url[len("http://") :]
        return client.get(url, params=params, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.base.requests.get",
        side_effect=patched_get,
    )


@pytest.fixture
def request_post_mock(mocker, client):
    def patched_post(url, data=None, json=None, **kwargs):
        url = url[len("http://") :]
        return client.post(url, data=data, json=json, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.base.requests.post",
        side_effect=patched_post,
    )


@pytest.mark.usefixtures("request_get_mock", "request_post_mock")
class DeployStateMock(DeployState):
    def get_client(self) -> BaseClient:
        return HTTPClient(host="", port=None)


class DeployMetaMock(DeployMeta):
    class Config:
        use_enum_values = True

    type: ClassVar = "mock"
    status: DeployStatus = DeployStatus.NOT_DEPLOYED
    param: str = ""
    state: DeployState = DeployStateMock()


class TargetEnvMock(TargetEnvMeta):
    type: ClassVar = "mock"
    deploy_type: ClassVar = DeployMetaMock

    def deploy(self, meta: DeployMetaMock):
        meta.status = DeployStatus.RUNNING
        meta.update()

    def destroy(self, meta: DeployMetaMock):
        meta.status = DeployStatus.STOPPED
        meta.update()

    def get_status(
        self, meta: DeployMetaMock, raise_on_error=True
    ) -> "DeployStatus":
        return meta.status


@pytest.fixture
def mock_env_path(tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "mock-target-env")
    TargetEnvMock().dump(path)
    return path


@pytest.fixture()
def mock_deploy_path(tmp_path, mock_env_path, model_meta_saved_single):
    path = os.path.join(tmp_path, "deployname")
    DeployMetaMock(
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
        f"deploy create {path} -m {model_meta_saved_single.loc.uri} -t {mock_env_path} -c param=aaa".split()
    )
    assert result.exit_code == 0, result.output
    assert os.path.isfile(path + MLEM_EXT)
    meta = load_meta(path)
    assert isinstance(meta, DeployMetaMock)
    assert meta.param == "aaa"
    assert meta.status == DeployStatus.RUNNING


def test_deploy_create_existing(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy create {mock_deploy_path}".split())
    assert result.exit_code == 0, result.output
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, DeployMetaMock)
    assert meta.param == "bbb"
    assert meta.status == DeployStatus.RUNNING


def test_deploy_status(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy status {mock_deploy_path}".split())
    assert result.exit_code == 0, result.output
    assert result.output.strip() == DeployStatus.NOT_DEPLOYED.value


def test_deploy_destroy(runner: Runner, mock_deploy_path):
    result = runner.invoke(f"deploy teardown {mock_deploy_path}".split())
    assert result.exit_code == 0, result.output
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, DeployMetaMock)
    assert meta.status == DeployStatus.STOPPED


def test_deploy_apply(runner: Runner, mock_deploy_path, data_path):
    result = runner.invoke(
        f"deploy apply {mock_deploy_path} {data_path}".split()
    )
    assert result.exit_code == 0, result.output
