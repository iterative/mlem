import contextlib
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
    heroku_api_request,
)
from mlem.core.errors import DeploymentError
from mlem.core.objects import DeployStatus, ModelMeta
from tests.conftest import long

heroku = pytest.mark.skipif(
    HEROKU_CONFIG.API_KEY is None, reason="No HEROKU_API_KEY env provided"
)

HEROKU_TEST_APP_NAME = "mlem-test"
HEROKU_TEST_REAL_APP_NAME = "mlem-test2"
HEROKU_TEAM = os.environ.get("HEROKU_TEAM")


@contextlib.contextmanager
def heroku_app_name(name):
    name = f"{name}-{getpass.getuser()}"
    try:
        delete_app(name)
    except DeploymentError:
        pass

    yield name
    try:
        delete_app(name)
    except DeploymentError:
        pass


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


@pytest.fixture()
def heroku_deploy(heroku_env: HerokuEnvMeta, model):
    with heroku_app_name(HEROKU_TEST_APP_NAME) as name:
        yield HerokuDeploy(
            app_name=name,
            env_link=heroku_env.make_link(),
            model_link=model.make_link(),
            team=HEROKU_TEAM,
        )


@heroku
@long
def test_heroku_api_request():
    res = heroku_api_request("GET", "/schema", api_key="_")
    assert "$schema" in res


@heroku
@long
def test_create_app(heroku_deploy):
    create_app(heroku_deploy)
    assert heroku_api_request("GET", f"/apps/{heroku_deploy.app_name}")


@long
def test_build_heroku_docker(model: ModelMeta):
    image_meta = build_heroku_docker(model, HEROKU_TEST_APP_NAME, push=False)
    client = DockerClient.from_env()
    image = client.images.get(image_meta.image_id)
    assert image is not None
    assert (
        f"{DEFAULT_HEROKU_REGISTRY}/{HEROKU_TEST_APP_NAME}/web:latest"
        in image.tags
    )


def test_state_ensured_app():
    state = HerokuState()
    with pytest.raises(ValueError):
        assert state.ensured_app is not None

    state.app = HerokuAppMeta(
        name=HEROKU_TEST_APP_NAME, web_url="", meta_info={}
    )

    assert state.ensured_app.name == HEROKU_TEST_APP_NAME


@pytest.fixture()
def real_app():
    with heroku_app_name(HEROKU_TEST_REAL_APP_NAME) as name:
        yield name


@heroku
@long
def test_env_deploy_new(tmp_path_factory, model, heroku_env, real_app):
    meta_path = tmp_path_factory.mktemp("deploy-meta")
    meta = deploy(str(meta_path), model, heroku_env, app_name=real_app)

    assert isinstance(meta, HerokuDeploy)
    assert heroku_api_request("GET", f"/apps/{meta.state.ensured_app.name}")
    meta.wait_for_status(
        DeployStatus.RUNNING,
        allowed_intermediate=DeployStatus.STARTING,
        times=15,
    )
    assert meta.get_status() == DeployStatus.RUNNING
    time.sleep(1)
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

    meta.destroy()

    assert meta.state is None
    meta.wait_for_status(
        DeployStatus.NOT_DEPLOYED,
        allowed_intermediate=DeployStatus.RUNNING,
        times=15,
    )
    with pytest.raises(DeploymentError):
        delete_app(real_app)
