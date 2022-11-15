import getpass
import os
import time

import pytest
import requests
from docker import DockerClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from mlem.api.commands import deploy
from mlem.contrib.heroku.build import (
    DEFAULT_HEROKU_REGISTRY,
    build_heroku_docker,
)
from mlem.contrib.heroku.config import HEROKU_CONFIG
from mlem.contrib.heroku.meta import (
    HerokuAppMeta,
    HerokuDeployment,
    HerokuEnv,
    HerokuState,
)
from mlem.contrib.heroku.utils import (
    create_app,
    delete_app,
    get_app,
    heroku_api_request,
)
from mlem.core.errors import DeploymentError
from mlem.core.objects import DeployStatus, MlemModel
from tests.conftest import flaky, long

heroku = pytest.mark.skipif(
    HEROKU_CONFIG.API_KEY is None, reason="No HEROKU_API_KEY env provided"
)
HEROKU_TEST_APP_NAME_PREFIX = "mlem-test"
CLEAR_APPS = False
HEROKU_TEAM = os.environ.get("HEROKU_TEAM")


@pytest.fixture()
def heroku_app_name():
    container = []

    def inner(prefix):
        name = f"{HEROKU_TEST_APP_NAME_PREFIX}-{prefix}-{getpass.getuser()}"

        if get_app(name, HEROKU_TEAM) is not None:
            delete_app(name)

        container.append(name)
        return name

    yield inner
    if CLEAR_APPS:
        for item in container:
            if get_app(item, HEROKU_TEAM) is not None:
                delete_app(item)


@pytest.fixture()
def heroku_env(tmpdir_factory):
    return HerokuEnv().dump(str(tmpdir_factory.mktemp("heroku_test") / "env"))


@pytest.fixture()
def model(tmpdir_factory):
    # TODO: change after https://github.com/iterative/mlem/issues/158
    # model = MlemModel.from_obj(lambda x: x, sample_data=[1, 2])
    bulk = load_iris(as_frame=True)
    X, y = bulk.data, bulk.target  # pylint: disable=no-member
    clf = RandomForestClassifier(n_estimators=1)
    clf.fit(X, y)
    model = MlemModel.from_obj(clf, sample_data=X)
    return model.dump(str(tmpdir_factory.mktemp("heroku_test") / "model"))


@heroku
@long
def test_heroku_api_request():
    res = heroku_api_request("GET", "/schema", api_key="_")
    assert "$schema" in res


@heroku
@long
def test_create_app(heroku_app_name, heroku_env, model):
    name = heroku_app_name("create-app")
    heroku_deploy = HerokuDeployment(
        app_name=name,
        env=heroku_env,
        model=model.make_link(),
        team=HEROKU_TEAM,
    )
    create_app(heroku_deploy)
    assert heroku_api_request("GET", f"/apps/{heroku_deploy.app_name}")


@long
def test_build_heroku_docker(model: MlemModel, uses_docker_build):
    image_meta = build_heroku_docker(model, "test_build", push=False)
    client = DockerClient.from_env()
    image = client.images.get(image_meta.image_id)
    assert image is not None
    assert f"{DEFAULT_HEROKU_REGISTRY}/test_build/web:latest" in image.tags


def test_state_ensured_app():
    state = HerokuState(declaration=HerokuDeployment(app_name=""))
    with pytest.raises(ValueError):
        assert state.ensured_app is not None

    state.app = HerokuAppMeta(name="name", web_url="", meta_info={})

    assert state.ensured_app.name == "name"


def _check_heroku_deployment(meta):
    assert isinstance(meta, HerokuDeployment)
    state = meta.get_state()
    assert heroku_api_request("GET", f"/apps/{state.ensured_app.name}")
    meta.wait_for_status(
        DeployStatus.RUNNING,
        allowed_intermediate=[
            DeployStatus.NOT_DEPLOYED,
            DeployStatus.STARTING,
        ],
        times=50,
    )
    assert meta.get_status() == DeployStatus.RUNNING
    time.sleep(10)
    docs_page = requests.post(
        state.ensured_app.web_url + "predict",
        json={
            "data": {
                "values": [
                    {
                        "sepal length (cm)": 0,
                        "sepal width (cm)": 0,
                        "petal length (cm)": 0,
                        "petal width (cm)": 0,
                    }
                ]
            }
        },
    )
    docs_page.raise_for_status()
    res = docs_page.json()
    assert isinstance(res, list)
    assert len(res) == 1


def is_not_crash(err, *args):  # pylint: disable=unused-argument
    needs_another_try = issubclass(err[0], DeploymentError)
    if needs_another_try:
        time.sleep(10)
    return not needs_another_try


@flaky(rerun_filter=is_not_crash, max_runs=1)
@heroku
@long
def test_env_deploy_full(
    tmp_path_factory,
    model: MlemModel,
    heroku_env,
    heroku_app_name,
    uses_docker_build,
):
    name = heroku_app_name("full-cycle")
    meta_path = tmp_path_factory.mktemp("deploy-meta")
    meta = deploy(
        str(meta_path), model, heroku_env, app_name=name, team=HEROKU_TEAM
    )

    _check_heroku_deployment(meta)

    model.params = {"version": "new"}
    model.update()
    redeploy_meta = deploy(meta, model, heroku_env)

    _check_heroku_deployment(redeploy_meta)
    if CLEAR_APPS:
        meta.remove()

        assert meta.get_state() == HerokuState(declaration=meta)
        meta.wait_for_status(
            DeployStatus.NOT_DEPLOYED,
            allowed_intermediate=DeployStatus.RUNNING,
            times=15,
        )
        with pytest.raises(DeploymentError):
            delete_app(name)
