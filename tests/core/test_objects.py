import os
import tempfile
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from pydantic import ValidationError
from sklearn.datasets import load_iris

from mlem.core.artifacts import LocalArtifact
from mlem.core.errors import MlemRootNotFound
from mlem.core.meta_io import (
    ART_DIR,
    META_FILE_NAME,
    MLEM_DIR,
    MLEM_EXT,
    deserialize,
    serialize,
)
from mlem.core.metadata import load, load_meta
from mlem.core.objects import (
    Deployment,
    DeployMeta,
    LinkData,
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


@pytest.fixture(params=["fullpath", "with_root"])
def path_and_root(mlem_repo, request):
    def get(name):
        if request.param == "fullpath":
            return os.path.join(mlem_repo, name), None
        return name, mlem_repo

    return get


@pytest.mark.parametrize("external", [True, False])
def test_meta_dump_curdir(meta, mlem_curdir_repo, external):
    meta.dump(DEPLOY_NAME, external=external)
    path = DEPLOY_NAME + MLEM_EXT
    if not external:
        path = os.path.join(MLEM_DIR, meta.object_type, path)
    assert os.path.isfile(path)
    assert isinstance(load(DEPLOY_NAME), DeployMeta)


def test_meta_dump__no_root(meta, tmpdir):
    with pytest.raises(MlemRootNotFound):
        meta.dump(DEPLOY_NAME, repo=str(tmpdir))


def test_meta_dump_internal(mlem_repo, meta, path_and_root):
    path, root = path_and_root(DEPLOY_NAME)
    meta.dump(path, repo=root, external=False)
    meta_path = os.path.join(
        mlem_repo, MLEM_DIR, DeployMeta.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(meta_path)
    assert isinstance(load_meta(meta_path), DeployMeta)
    assert isinstance(load_meta(path, repo=root), DeployMeta)


def test_meta_dump_external(mlem_repo, meta, path_and_root):
    path, root = path_and_root(DEPLOY_NAME)
    meta.dump(path, repo=root, external=True)
    meta_path = os.path.join(mlem_repo, DEPLOY_NAME + MLEM_EXT)
    assert os.path.isfile(meta_path)
    assert isinstance(load_meta(meta_path), DeployMeta)
    link_path = os.path.join(
        mlem_repo, MLEM_DIR, MlemLink.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    assert isinstance(load_meta(link_path, follow_links=False), MlemLink)


@pytest.mark.parametrize("external", [False])
def test_model_dump_curdir(model_meta, mlem_curdir_repo, external):
    model_meta.dump(MODEL_NAME, external=external)
    if not external:
        prefix = Path(os.path.join(MLEM_DIR, model_meta.object_type))
    else:
        prefix = Path("")
    assert os.path.isdir(prefix / MODEL_NAME)
    assert os.path.isfile(prefix / MODEL_NAME / META_FILE_NAME)
    assert os.path.isdir(prefix / MODEL_NAME / ART_DIR)
    assert os.path.isfile(prefix / MODEL_NAME / ART_DIR / "data.pkl")
    assert isinstance(load_meta(MODEL_NAME), ModelMeta)


def test_model_dump_internal(mlem_repo, model_meta, path_and_root):
    path, root = path_and_root(MODEL_NAME)
    model_meta.dump(path, repo=root, external=False)
    model_path = os.path.join(
        mlem_repo, MLEM_DIR, ModelMeta.object_type, MODEL_NAME
    )
    assert os.path.isdir(model_path)
    assert os.path.isfile(os.path.join(model_path, META_FILE_NAME))
    assert os.path.exists(os.path.join(model_path, ART_DIR, "data.pkl"))


def test_model_dump_external(mlem_repo, model_meta, path_and_root):
    path, root = path_and_root(MODEL_NAME)
    model_meta.dump(path, repo=root, external=True)
    model_path = os.path.join(mlem_repo, MODEL_NAME)
    assert os.path.isdir(model_path)
    assert os.path.isfile(os.path.join(model_path, META_FILE_NAME))
    assert os.path.exists(os.path.join(model_path, ART_DIR, "data.pkl"))
    link_path = os.path.join(
        mlem_repo, MLEM_DIR, MlemLink.object_type, MODEL_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    link = load_meta(link_path, follow_links=False)
    assert isinstance(link, MlemLink)
    model = link.load_link()
    assert serialize(model) == serialize(model_meta)


def _check_cloned_model(cloned_model_meta: MlemMeta, path, fs=None):
    if fs is None:
        fs = LocalFileSystem()
    assert isinstance(cloned_model_meta, ModelMeta)
    assert cloned_model_meta.artifacts is not None
    for art in cloned_model_meta.artifacts:
        assert isinstance(art, LocalArtifact)
        assert not os.path.isabs(art.uri)
        assert fs.isfile(os.path.join(path, art.uri))
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
def test_model_cloning_to_remote(model_path, s3_tmp_path, s3_storage_fs):
    model = load_meta(model_path)
    path = s3_tmp_path("model_cloning_to_remote")
    model.clone(path, link=False)
    s3path = Path(path[len("s3:/") :])
    assert s3_storage_fs.isfile(s3path / META_FILE_NAME)
    assert s3_storage_fs.isdir(s3path / ART_DIR)
    assert s3_storage_fs.isfile(s3path / ART_DIR / "data.pkl")
    cloned_model_meta = load_meta(path, load_value=True)
    _check_cloned_model(cloned_model_meta, path, s3_storage_fs)


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


def test_mlem_dir_path(filled_mlem_repo):
    # case when we provide objects' abspath and object is already located in the same MLEM root
    model_link = os.path.join(
        filled_mlem_repo, MLEM_DIR, "model", "data", "model" + MLEM_EXT
    )
    assert (
        mlem_dir_path(
            os.path.join(filled_mlem_repo, "data", "model"),
            obj_type="model",
            fs=None,
        )
        == model_link
    )
    # case when we provide object relative path
    model_link = os.path.join(
        filled_mlem_repo, MLEM_DIR, "model", "latest" + MLEM_EXT
    )
    assert (
        mlem_dir_path(
            "latest", fs=None, obj_type="model", repo=filled_mlem_repo
        )
        == model_link
    )


def test_link_dump(model_path):
    link = MlemLink(
        link_data=LinkData(path=os.path.join(model_path, META_FILE_NAME)),
        link_type="model",
    )
    with tempfile.TemporaryDirectory() as dir:
        path_to_link = os.path.join(dir, "latest" + MLEM_EXT)
        link.dump(path_to_link)
        model = load_meta(path_to_link, follow_links=True)
    assert isinstance(model, ModelMeta)


def test_double_link_load(filled_mlem_repo):
    latest = load_meta("latest", repo=filled_mlem_repo, follow_links=False)
    link = latest.make_link("external", repo=filled_mlem_repo, external=True)
    assert link.link_type == "model"
    model = load_meta("external", repo=filled_mlem_repo, follow_links=True)
    assert isinstance(model, ModelMeta)


def test_link_dump_in_mlem(model_path_mlem_repo):
    model_path, mlem_repo = model_path_mlem_repo
    link = MlemLink(
        link_data=LinkData(path=os.path.join(model_path, META_FILE_NAME)),
        link_type="model",
    )
    link_name = "latest"
    link.dump(link_name, repo=mlem_repo, external=True, link=False)
    model = load_meta(os.path.join(mlem_repo, link_name), follow_links=True)
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


def test_mlem_repo_root(filled_mlem_repo):
    path = Path(filled_mlem_repo)
    assert os.path.exists(path)
    assert os.path.isdir(path)
    mlem_dir = path / MLEM_DIR
    assert os.path.isdir(mlem_dir)
    assert os.path.isfile(mlem_dir / "link" / ("model1" + MLEM_EXT))
    assert os.path.isfile(mlem_dir / "link" / ("latest" + MLEM_EXT))
    model_dir = path / "model1"
    assert os.path.isdir(model_dir)
    assert os.path.isfile(model_dir / META_FILE_NAME)
    assert os.path.isfile(model_dir / ART_DIR / "data.pkl")
