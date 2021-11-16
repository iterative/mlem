import os
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError
from sklearn.datasets import load_iris

from mlem.core.artifacts import LocalArtifact
from mlem.core.meta_io import (
    ART_DIR,
    META_FILE_NAME,
    MLEM_DIR,
    MLEM_EXT,
    deserialize,
    serialize,
)
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    Deployment,
    DeployMeta,
    MlemLink,
    MlemMeta,
    ModelMeta,
    mlem_dir_path,
)
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth

DEPLOY_NAME = "mydeploy"
MODEL_NAME = "decision_tree"


class MyDeployment(Deployment):
    def get_status(self):
        pass

    def destroy(self):
        pass


@pytest.fixture()
def meta():
    return DeployMeta(env_path="", model_path="", deployment=MyDeployment())


def test_meta_dump_internal(mlem_root, meta):
    meta.dump(DEPLOY_NAME, mlem_root=mlem_root, external=False)
    meta_path = os.path.join(
        mlem_root, MLEM_DIR, DeployMeta.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(meta_path)
    assert isinstance(load_meta(meta_path), DeployMeta)


def _check_external_meta(meta_path, mlem_root):
    assert os.path.isfile(meta_path)
    assert isinstance(load_meta(meta_path), DeployMeta)
    link_path = os.path.join(
        mlem_root, MLEM_DIR, DeployMeta.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    assert isinstance(load_meta(link_path, follow_links=False), MlemLink)


def test_meta_dump_external(mlem_root, meta):
    meta.dump(DEPLOY_NAME, mlem_root=mlem_root, external=True)
    meta_path = os.path.join(mlem_root, DEPLOY_NAME + MLEM_EXT)
    _check_external_meta(meta_path, mlem_root)


def test_meta_dump_external_fullpath(mlem_root, meta):
    meta_path = os.path.join(mlem_root, DEPLOY_NAME + MLEM_EXT)
    meta.dump(meta_path, external=True)
    _check_external_meta(meta_path, mlem_root)


def test_model_dump_internal(mlem_root, model_meta):
    model_meta.dump(MODEL_NAME, mlem_root=mlem_root, external=False)
    model_path = os.path.join(
        mlem_root, MLEM_DIR, ModelMeta.object_type, MODEL_NAME
    )
    assert os.path.isdir(model_path)
    assert os.path.isfile(os.path.join(model_path, META_FILE_NAME))
    assert os.path.exists(os.path.join(model_path, ART_DIR, "data.pkl"))


def _check_external_model(meta, mlem_root, model_path):
    assert os.path.isdir(model_path)
    assert os.path.isfile(os.path.join(model_path, META_FILE_NAME))
    assert os.path.exists(os.path.join(model_path, ART_DIR, "data.pkl"))
    link_path = os.path.join(
        mlem_root, MLEM_DIR, ModelMeta.object_type, MODEL_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    link = load_meta(link_path, follow_links=False)
    assert isinstance(link, MlemLink)
    model = link.load_link()
    assert serialize(model) == serialize(meta)


def test_model_dump_external(mlem_root, model_meta):
    model_meta.dump(MODEL_NAME, mlem_root=mlem_root, external=True)
    model_path = os.path.join(mlem_root, MODEL_NAME)
    _check_external_model(model_meta, mlem_root, model_path)


def test_model_dump_external_fullpath(mlem_root, model_meta):
    model_path = os.path.join(mlem_root, MODEL_NAME)
    model_meta.dump(model_path, external=True)
    _check_external_model(model_meta, mlem_root, model_path)


def _check_cloned_model(cloned_model_meta: MlemMeta, path):
    assert isinstance(cloned_model_meta, ModelMeta)
    assert cloned_model_meta.artifacts is not None
    for art in cloned_model_meta.artifacts:
        assert isinstance(art, LocalArtifact)
        assert not os.path.isabs(art.uri)
        assert os.path.isfile(os.path.join(path, art.uri))
    cloned_model = cloned_model_meta.get_value()
    assert cloned_model is not None
    X, _ = load_iris(return_X_y=True)
    cloned_model.predict(X)


def test_model_cloning(model_path):
    model = load_meta(model_path)
    with tempfile.TemporaryDirectory() as path:
        model.clone(path, link=False)
        cloned_model_meta = load_meta(path, load_value=True)
        _check_cloned_model(cloned_model_meta, path)


@long
@pytest.mark.xfail("TODO: https://github.com/iterative/mlem/issues/108")
def test_model_cloning_to_remote(model_path, s3_tmp_path):
    model = load_meta(model_path)
    path = s3_tmp_path("model_cloning_to_remote")
    model.clone(path, link=False)
    cloned_model_meta = load_meta(path, load_value=True)
    _check_cloned_model(cloned_model_meta, path)


@long
@need_test_repo_auth
def test_model_cloning_remote(current_test_branch):
    """TODO: https://github.com/iterative/mlem/issues/44
    test fails in CI because repo is private and DVC does not support http auth for git
    """
    with tempfile.TemporaryDirectory() as dir:
        cloned_model = load_meta(
            os.path.join(MLEM_TEST_REPO, "simple/data/model"),
            rev=current_test_branch,
        ).clone(os.path.join(dir, "model"), link=False)
        cloned_model.load_value()
        X, _ = load_iris(return_X_y=True)
        cloned_model.predict(X)


def test_model_getattr(model_meta):
    method = model_meta.predict
    assert callable(method)
    X, _ = load_iris(return_X_y=True)
    method(X)

    with pytest.raises(AttributeError):
        model_meta.not_existing_method(X)


def test_mlem_dir_path(filled_mlem_root):
    # case when we provide objects' abspath and object is already located in the same MLEM root
    model_link = os.path.join(
        filled_mlem_root, MLEM_DIR, "model", "data", "model" + MLEM_EXT
    )
    assert (
        mlem_dir_path(
            os.path.join(filled_mlem_root, "data", "model"),
            obj_type="model",
            fs=None,
        )
        == model_link
    )
    # case when we provide object relative path
    model_link = os.path.join(
        filled_mlem_root, MLEM_DIR, "model", "latest" + MLEM_EXT
    )
    assert (
        mlem_dir_path(
            "latest", fs=None, obj_type="model", mlem_root=filled_mlem_root
        )
        == model_link
    )


def test_link_dump(model_path):
    link = MlemLink(
        mlem_link=os.path.join(model_path, META_FILE_NAME), link_type="model"
    )
    with tempfile.TemporaryDirectory() as dir:
        path_to_link = os.path.join(dir, "latest" + MLEM_EXT)
        link.dump(path_to_link)
        model = load_meta(path_to_link, follow_links=True)
    assert isinstance(model, ModelMeta)


@pytest.mark.xfail
def test_double_link_load():
    raise AssertionError()


def test_link_dump_in_mlem(model_path_mlem_root):
    model_path, mlem_root = model_path_mlem_root
    link = MlemLink(
        mlem_link=os.path.join(model_path, META_FILE_NAME), link_type="model"
    )
    link_name = "latest"
    link.dump(link_name, mlem_root=mlem_root, external=True, link=False)
    model = load_meta(os.path.join(mlem_root, link_name), follow_links=True)
    assert isinstance(model, ModelMeta)


def test_model_model_type_laziness():
    payload = {
        "model_type": {"type": "doesnotexist"},
        "object_type": "model",
        "requirements": [],
    }
    model = deserialize(payload, ModelMeta)
    assert model.model_type_raw == {"type": "doesnotexist"}
    with pytest.raises(ValidationError):
        print(model.model_type)


def test_mlem_root(filled_mlem_root):
    path = Path(filled_mlem_root)
    assert os.path.exists(path)
    assert os.path.isdir(path)
    mlem_dir = path / MLEM_DIR
    assert os.path.isdir(mlem_dir)
    assert os.path.isfile(mlem_dir / "model" / ("model1" + MLEM_EXT))
    assert os.path.isfile(mlem_dir / "model" / ("latest" + MLEM_EXT))
    model_dir = path / "model1"
    assert os.path.isdir(model_dir)
    assert os.path.isfile(model_dir / META_FILE_NAME)
    assert os.path.isfile(model_dir / ART_DIR / "data.pkl")
