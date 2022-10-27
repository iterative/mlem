import os
from typing import Any, ClassVar, Optional, Type

import pytest
from numpy import ndarray
from yaml import safe_load

from mlem.api import load
from mlem.cli.declare import create_declare_mlem_object_subcommand, declare
from mlem.cli.deployment import create_deploy_run_command
from mlem.contrib.heroku.meta import HerokuEnv
from mlem.core.errors import DeploymentError, WrongMetaSubType
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemEnv,
    MlemLink,
    MlemModel,
    MlemObject,
)
from mlem.runtime.client import Client, HTTPClient
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


class DeployStateMock(DeployState):
    """mock"""

    class Config:
        use_enum_values = True

    allow_default: ClassVar = True

    deployment: Optional[MlemDeployment] = None
    env: Optional[MlemEnv] = None
    status: DeployStatus = DeployStatus.NOT_DEPLOYED


class MlemEnvMock(MlemEnv):
    """mock"""

    type: ClassVar = "mock"

    env_param: Optional[str] = None


class MlemDeploymentMock(MlemDeployment[DeployStateMock, MlemEnvMock]):
    """mock"""

    type: ClassVar = "mock"
    state_type: ClassVar = DeployStateMock
    env_type: ClassVar = MlemEnvMock

    """status"""
    param: str = ""
    """param"""

    def _get_client(self, state) -> Client:
        return HTTPClient(host="", port=None)

    def deploy(self, model: MlemModel):
        with self.lock_state():
            state = self.get_state()
            state.status = DeployStatus.RUNNING
            state.deployment = self
            state.env = self.get_env()
            state.update_model(model)
            self.update_state(state)

    def remove(self):
        with self.lock_state():
            state = self.get_state()
            state.status = DeployStatus.STOPPED
            state.deployment = None
            state.env = None
            state.model_hash = None
            self.update_state(state)

    def get_status(self, raise_on_error=True) -> "DeployStatus":
        with self.lock_state():
            return self.get_state().status


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
        f"deploy run {MlemDeploymentMock.type} {path} -m {model_meta_saved_single.loc.uri} --env {mock_env_path} --param aaa".split()
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
    assert meta.get_status() == DeployStatus.RUNNING


def test_deploy_create_existing(
    runner: Runner, mock_deploy_path, model_meta_saved_single
):
    result = runner.invoke(
        f"deploy run --load {mock_deploy_path} -m {model_meta_saved_single.loc.fullpath}".split(),
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
    assert meta.get_status() == DeployStatus.RUNNING


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
    assert meta.get_status() == DeployStatus.STOPPED


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
        f"deploy apply {mock_deploy_path} {data_path} -o {path}".split(),
        raise_on_error=True,
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    meta = load_meta(mock_deploy_path)
    assert isinstance(meta, MlemDeploymentMock)
    assert meta.get_status() == DeployStatus.NOT_DEPLOYED
    predictions = load(path)
    assert isinstance(predictions, ndarray)


def add_mock_declare(type_: Type[MlemObject]):

    typer = [
        g.typer_instance
        for g in declare.registered_groups
        if g.typer_instance.info.name == type_.object_type
    ][0]

    create_declare_mlem_object_subcommand(
        typer,
        type_.__get_alias__(),
        type_.object_type,
        type_,
    )


add_mock_declare(MlemDeploymentMock)
add_mock_declare(MlemEnvMock)

create_deploy_run_command(MlemDeploymentMock.type)


def _deploy_and_check(
    runner: Runner,
    deploy_path: str,
    model_single_path: str,
    load_deploy=True,
    add_args="",
    env_param_value: Optional[str] = "env_val",
):

    if load_deploy:
        status_res = runner.invoke(
            f"deploy status {deploy_path}", raise_on_error=True
        )
        assert status_res.exit_code == 0, (
            status_res.output,
            status_res.exception,
            status_res.stderr,
        )
        assert status_res.output.strip() == DeployStatus.NOT_DEPLOYED.value

        deploy_res = runner.invoke(
            f"deploy run --load {deploy_path} --model {model_single_path}",
            raise_on_error=True,
        )
    else:
        deploy_res = runner.invoke(
            f"deploy run {MlemDeploymentMock.type} {deploy_path} --model {model_single_path} --param val {add_args}",
            raise_on_error=True,
        )

    assert deploy_res.exit_code == 0, (
        deploy_res.output,
        deploy_res.exception,
        deploy_res.stderr,
    )

    status_res = runner.invoke(
        f"deploy status {deploy_path}", raise_on_error=True
    )
    assert status_res.exit_code == 0, (
        status_res.output,
        status_res.exception,
        status_res.stderr,
    )
    assert status_res.output.strip() == DeployStatus.RUNNING.value

    deploy_meta = load_meta(deploy_path, force_type=MlemDeploymentMock)
    state = deploy_meta.get_state()
    assert isinstance(state.deployment, MlemDeploymentMock)
    assert state.deployment.param == "val"
    assert isinstance(state.env, MlemEnvMock)
    assert state.env.env_param == env_param_value

    remove_res = runner.invoke(
        f"deploy remove {deploy_path}", raise_on_error=True
    )
    assert remove_res.exit_code == 0, (
        remove_res.output,
        remove_res.exception,
        remove_res.stderr,
    )

    status_res = runner.invoke(
        f"deploy status {deploy_path}", raise_on_error=True
    )
    assert status_res.exit_code == 0, (
        status_res.output,
        status_res.exception,
        status_res.stderr,
    )
    assert status_res.output.strip() == DeployStatus.STOPPED.value


def test_all_declared(runner: Runner, tmp_path, model_single_path):
    """
    mlem declare env heroku --api_key lol prod.mlem
    mlem declare deployment heroku --env prod.mlem --app_name myapp service.mlem
    # error on depl/env type mismatch  TODO
    mlem deployment run --load service.mlem --model mdoel
    """
    env_path = make_posix(str(tmp_path / "env"))
    runner.invoke(
        f"declare env {MlemEnvMock.type} --env_param env_val {env_path}",
        raise_on_error=True,
    )
    deploy_path = make_posix(str(tmp_path / "deploy"))
    runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} --param val --env {env_path} {deploy_path}",
        raise_on_error=True,
    )

    _deploy_and_check(runner, deploy_path, model_single_path)


def test_declare_type_mismatch(runner: Runner, tmp_path, model_single_path):
    """
    mlem declare env heroku --api_key lol prod.mlem
    mlem declare deployment sagemaker --env prod.mlem --app_name myapp service.mlem
    # error on depl/env type mismatch  TODO
    mlem deployment run --load service.mlem --model mdoel
    """
    env_path = make_posix(str(tmp_path / "env"))
    runner.invoke(
        f"declare env {HerokuEnv.type} {env_path}", raise_on_error=True
    )
    deploy_path = make_posix(str(tmp_path / "deploy"))
    runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} --param a --env {env_path} {deploy_path}",
        raise_on_error=True,
    )

    with pytest.raises(WrongMetaSubType):
        runner.invoke(
            f"deploy run --load {deploy_path} --model {model_single_path}",
            raise_on_error=True,
        )


def test_deploy_declared(runner: Runner, tmp_path, model_single_path):
    """
    mlem declare deployment heroku --env.api_key prod.mlem --app_name myapp service.mlem
    mlem deployment run --load service.mlem --model mdoel
    """
    deploy_path = make_posix(str(tmp_path / "deploy"))
    declare_res = runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} {deploy_path} --param val --env.env_param env_val ",
        raise_on_error=True,
    )
    assert declare_res.exit_code == 0, (
        declare_res.output,
        declare_res.exception,
        declare_res.stderr,
    )

    _deploy_and_check(runner, deploy_path, model_single_path)


def test_env_declared(runner: Runner, tmp_path, model_single_path):
    """
    mlem declare env heroku --api_key lol prod.mlem
    mlem deployment run heroku service.mlem --model model --app_name myapp --env prod.mlem
    # error on type mismatch
    """
    env_path = make_posix(str(tmp_path / "env"))
    declare_res = runner.invoke(
        f"declare env {MlemEnvMock.type} --env_param env_val {env_path}",
        raise_on_error=True,
    )
    assert declare_res.exit_code == 0, (
        declare_res.output,
        declare_res.exception,
        declare_res.stderr,
    )
    deploy_path = make_posix(str(tmp_path / "deploy"))
    _deploy_and_check(
        runner,
        deploy_path,
        model_single_path,
        load_deploy=False,
        add_args=f"--env {env_path}",
    )


def test_none_declared(runner: Runner, tmp_path, model_single_path):
    """
    mlem deployment run heroku service.mlem --model model --app_name myapp --env.api_key lol
    # error on args mismatch
    """
    deploy_path = make_posix(str(tmp_path / "deploy"))
    _deploy_and_check(
        runner,
        deploy_path,
        model_single_path,
        load_deploy=False,
        add_args="--env.env_param env_val",
    )


def test_no_env_params(runner: Runner, tmp_path, model_single_path):
    deploy_path = make_posix(str(tmp_path / "deploy"))
    _deploy_and_check(
        runner,
        deploy_path,
        model_single_path,
        load_deploy=False,
        env_param_value=None,
    )


def test_redeploy_changed(runner: Runner, tmp_path, model_single_path):
    env_path = make_posix(str(tmp_path / "env"))
    runner.invoke(
        f"declare env {MlemEnvMock.type} --env_param env_val {env_path}",
        raise_on_error=True,
    )
    deploy_path = make_posix(str(tmp_path / "deploy"))
    runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} --param val --env {env_path} {deploy_path}",
        raise_on_error=True,
    )

    runner.invoke(
        f"deploy run --load {deploy_path} --model {model_single_path}",
        raise_on_error=True,
    )

    runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} --param val1 --env {env_path} {deploy_path}",
        raise_on_error=True,
    )
    with pytest.raises(DeploymentError):
        runner.invoke(
            f"deploy run --load {deploy_path} --model {model_single_path}",
            raise_on_error=True,
        )


def test_redeploy_env_changed(runner: Runner, tmp_path, model_single_path):
    env_path = make_posix(str(tmp_path / "env"))
    runner.invoke(
        f"declare env {MlemEnvMock.type} --env_param env_val {env_path}",
        raise_on_error=True,
    )
    deploy_path = make_posix(str(tmp_path / "deploy"))
    runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} --param val --env {env_path} {deploy_path}",
        raise_on_error=True,
    )

    runner.invoke(
        f"deploy run --load {deploy_path} --model {model_single_path}",
        raise_on_error=True,
    )

    runner.invoke(
        f"declare env {MlemEnvMock.type} --env_param env_val1 {env_path}",
        raise_on_error=True,
    )

    with pytest.raises(DeploymentError):
        runner.invoke(
            f"deploy run --load {deploy_path} --model {model_single_path}",
            raise_on_error=True,
        )
