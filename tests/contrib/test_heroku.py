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
    HerokuDeploy,
    HerokuEnvMeta,
    HerokuState,
)
from mlem.contrib.heroku.utils import (
    create_app,
    delete_app,
    get_app,
    heroku_api_request,
)
from mlem.core.errors import DeploymentError
from mlem.core.objects import DeployStatus, ModelMeta
from tests.conftest import long, skip_matrix

heroku = pytest.mark.skipif(
    HEROKU_CONFIG.API_KEY is None, reason="No HEROKU_API_KEY env provided"
)
heroku_matrix = skip_matrix("ubuntu-latest", "3.7")
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
    return HerokuEnvMeta().dump(
        str(tmpdir_factory.mktemp("heroku_test") / "env")
    )


@pytest.fixture()
def model(tmpdir_factory):
    # TODO: change after https://github.com/iterative/mlem/issues/158
    # model = ModelMeta.from_obj(lambda x: x, sample_data=[1, 2])
    bulk = load_iris(as_frame=True)
    X, y = bulk.data, bulk.target  # pylint: disable=no-member
    clf = RandomForestClassifier(n_estimators=1)
    clf.fit(X, y)
    model = ModelMeta.from_obj(clf, sample_data=X)
    return model.dump(str(tmpdir_factory.mktemp("heroku_test") / "model"))


@heroku
@long
def test_heroku_api_request():
    res = heroku_api_request("GET", "/schema", api_key="_")
    assert "$schema" in res


@heroku
@long
@heroku_matrix
def test_create_app(heroku_app_name, heroku_env, model):
    name = heroku_app_name("create-app")
    heroku_deploy = HerokuDeploy(
        app_name=name,
        env_link=heroku_env.make_link(),
        model_link=model.make_link(),
        team=HEROKU_TEAM,
    )
    create_app(heroku_deploy)
    assert heroku_api_request("GET", f"/apps/{heroku_deploy.app_name}")


@long
@heroku_matrix
def test_build_heroku_docker(model: ModelMeta, uses_docker_build):
    image_meta = build_heroku_docker(model, "test_build", push=False)
    client = DockerClient.from_env()
    image = client.images.get(image_meta.image_id)
    assert image is not None
    assert f"{DEFAULT_HEROKU_REGISTRY}/test_build/web:latest" in image.tags


def test_state_ensured_app():
    state = HerokuState()
    with pytest.raises(ValueError):
        assert state.ensured_app is not None

    state.app = HerokuAppMeta(name="name", web_url="", meta_info={})

    assert state.ensured_app.name == "name"


@heroku
@long
@heroku_matrix
def test_env_deploy_full(
    tmp_path_factory, model, heroku_env, heroku_app_name, uses_docker_build
):
    name = heroku_app_name("full-cycle")
    meta_path = tmp_path_factory.mktemp("deploy-meta")
    meta = deploy(
        str(meta_path), model, heroku_env, app_name=name, team=HEROKU_TEAM
    )

    assert isinstance(meta, HerokuDeploy)
    assert heroku_api_request("GET", f"/apps/{meta.state.ensured_app.name}")
    meta.wait_for_status(
        DeployStatus.RUNNING,
        allowed_intermediate=[
            DeployStatus.NOT_DEPLOYED,
            DeployStatus.STARTING,
        ],
        times=25,
    )
    assert meta.get_status() == DeployStatus.RUNNING
    time.sleep(10)
    docs_page = requests.post(
        meta.state.ensured_app.web_url + "predict",
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

    if CLEAR_APPS:
        meta.destroy()

        assert meta.state is None
        meta.wait_for_status(
            DeployStatus.NOT_DEPLOYED,
            allowed_intermediate=DeployStatus.RUNNING,
            times=15,
        )
        with pytest.raises(DeploymentError):
            delete_app(name)
