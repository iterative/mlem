import os
import pickle
import posixpath
import tempfile
from pathlib import Path

import pytest
from fsspec.implementations.local import LocalFileSystem
from pydantic import ValidationError, parse_obj_as
from sklearn.datasets import load_iris

from mlem.core.artifacts import Artifacts, LocalArtifact, Storage
from mlem.core.errors import MlemRootNotFound
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.metadata import load, load_meta
from mlem.core.model import ModelIO
from mlem.core.objects import (
    DeployMeta,
    DeployState,
    MlemLink,
    MlemMeta,
    ModelMeta,
    mlem_dir_path,
)
from tests.conftest import (
    MLEM_TEST_REPO,
    long,
    need_test_repo_auth,
    need_test_repo_ssh_auth,
)

REV_LINK_TAG = "first_rev_link"

DEPLOY_NAME = "d/mydeploy"
MODEL_NAME = "m/decision_tree"


class MyDeployState(DeployState):
    def get_status(self):
        pass

    def destroy(self):
        pass


@pytest.fixture()
def meta():
    return DeployMeta(
        env_link=MlemLink(path="", link_type="env"),
        model_link=MlemLink(path="", link_type="model"),
        state=MyDeployState(),
    )


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


def test_meta_dump_fullpath_in_repo_no_link(mlem_repo, meta):
    meta.dump(
        os.path.join(mlem_repo, MLEM_DIR, meta.object_type, DEPLOY_NAME),
        link=True,
        external=True,
    )
    link_path = os.path.join(
        mlem_repo, MLEM_DIR, MlemLink.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert not os.path.exists(link_path)


def test_meta_dump_internal(mlem_repo, meta, path_and_root):
    path, root = path_and_root(DEPLOY_NAME)
    meta.dump(path, repo=root, external=False)
    assert meta.name == DEPLOY_NAME
    meta_path = os.path.join(
        mlem_repo, MLEM_DIR, DeployMeta.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(meta_path)
    load_path = load_meta(meta_path)
    assert isinstance(load_path, DeployMeta)
    assert load_path.name == meta.name
    load_root = load_meta(path, repo=root)
    assert isinstance(load_root, DeployMeta)
    assert load_root.name == meta.name


def test_meta_dump_external(mlem_repo, meta, path_and_root):
    path, root = path_and_root(DEPLOY_NAME)
    meta.dump(path, repo=root, external=True)
    assert meta.name == DEPLOY_NAME
    meta_path = os.path.join(mlem_repo, DEPLOY_NAME + MLEM_EXT)
    assert os.path.isfile(meta_path)
    loaded = load_meta(meta_path)
    assert isinstance(loaded, DeployMeta)
    assert loaded.name == meta.name
    link_path = os.path.join(
        mlem_repo, MLEM_DIR, MlemLink.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    assert isinstance(load_meta(link_path, follow_links=False), MlemLink)


@pytest.mark.parametrize("external", [False, True])
def test_model_dump_curdir(model_meta, mlem_curdir_repo, external):
    model_meta.dump(MODEL_NAME, external=external)
    assert model_meta.name == MODEL_NAME
    if not external:
        prefix = Path(os.path.join(MLEM_DIR, model_meta.object_type))
    else:
        prefix = Path("")
    assert os.path.isfile(prefix / MODEL_NAME)
    assert os.path.isfile(prefix / (MODEL_NAME + MLEM_EXT))
    assert isinstance(load_meta(MODEL_NAME), ModelMeta)


def test_model_dump_internal(mlem_repo, model_meta, path_and_root):
    path, root = path_and_root(MODEL_NAME)
    model_meta.dump(path, repo=root, external=False)
    assert model_meta.name == MODEL_NAME
    model_path = os.path.join(
        mlem_repo, MLEM_DIR, ModelMeta.object_type, MODEL_NAME
    )
    assert os.path.isfile(model_path + MLEM_EXT)
    assert os.path.isfile(model_path)


def test_model_dump_external(mlem_repo, model_meta, path_and_root):
    path, root = path_and_root(MODEL_NAME)
    model_meta.dump(path, repo=root, external=True)
    assert model_meta.name == MODEL_NAME
    model_path = os.path.join(mlem_repo, MODEL_NAME)
    assert os.path.isfile(model_path + MLEM_EXT)
    assert os.path.isfile(model_path)
    link_path = os.path.join(
        mlem_repo, MLEM_DIR, MlemLink.object_type, MODEL_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    link = load_meta(link_path, follow_links=False)
    assert isinstance(link, MlemLink)
    model = link.load_link()
    assert model.dict() == model_meta.dict()


def _check_cloned_model(cloned_model_meta: MlemMeta, path, fs=None):
    if fs is None:
        fs = LocalFileSystem()
    assert isinstance(cloned_model_meta, ModelMeta)
    assert cloned_model_meta.artifacts is not None
    assert len(cloned_model_meta.artifacts) == 1
    art = cloned_model_meta.artifacts[cloned_model_meta.model_type.io.art_name]
    assert isinstance(art, LocalArtifact)
    assert art.hash != ""
    assert art.size > 0
    assert art.uri == posixpath.basename(cloned_model_meta.name)
    assert not os.path.isabs(art.uri)
    assert fs.isfile(path)
    cloned_model_meta.load_value()
    cloned_model = cloned_model_meta.get_value()
    assert cloned_model is not None
    X, _ = load_iris(return_X_y=True)
    cloned_model.predict(X)


def _check_complex_cloned_model(cloned_model_meta: MlemMeta, path, fs=None):
    if fs is None:
        fs = LocalFileSystem()
    assert isinstance(cloned_model_meta, ModelMeta)
    assert cloned_model_meta.artifacts is not None
    assert len(cloned_model_meta.artifacts) == 2
    for name, art in cloned_model_meta.artifacts.items():
        assert isinstance(art, LocalArtifact)
        assert art.hash == ""
        assert art.size == 1
        assert art.uri == posixpath.join(
            posixpath.basename(cloned_model_meta.name), name
        )
        assert not os.path.isabs(art.uri)
        filepath = posixpath.join(path, name)
        assert fs.isfile(filepath)
        assert (
            fs.open(filepath).read().decode("utf8") == f"data{int(name[-1])}"
        )
    assert fs.isdir(path)


def test_model_cloning(model_single_path):
    model = load_meta(model_single_path)
    with tempfile.TemporaryDirectory() as path:
        path = posixpath.join(path, "cloned")
        model.clone(path, link=False)
        cloned_model_meta = load_meta(path, load_value=False)
        _check_cloned_model(cloned_model_meta, path)


def test_complex_model_cloning(complex_model_single_path):
    model = load_meta(complex_model_single_path)
    with tempfile.TemporaryDirectory() as path:
        path = posixpath.join(path, "cloned")
        model.clone(path, link=False)
        cloned_model_meta = load_meta(path, load_value=False)
        _check_complex_cloned_model(cloned_model_meta, path)


@pytest.mark.parametrize("external", [True, False])
def test_model_cloning_to_repo(model_single_path, mlem_repo, external):
    model = load_meta(model_single_path)
    model.clone("model", repo=mlem_repo, link=False, external=external)
    cloned_model_meta = load_meta("model", repo=mlem_repo, load_value=False)
    if external:
        path = os.path.join(mlem_repo, "model")
    else:
        path = os.path.join(
            mlem_repo, MLEM_DIR, ModelMeta.object_type, "model"
        )
    _check_cloned_model(cloned_model_meta, path)


@long
def test_model_cloning_to_remote(model_path, s3_tmp_path, s3_storage_fs):
    model = load_meta(model_path)
    path = s3_tmp_path("model_cloning_to_remote")
    model.clone(path, link=False)
    s3path = path[len("s3:/") :]
    assert s3_storage_fs.isfile(s3path + MLEM_EXT)
    assert s3_storage_fs.isfile(s3path)
    cloned_model_meta = load_meta(path, load_value=False)
    _check_cloned_model(cloned_model_meta, path, s3_storage_fs)


@pytest.fixture
def remote_model_meta(current_test_branch):
    def get(repo="simple"):
        return load_meta(
            os.path.join(MLEM_TEST_REPO, repo, "data", "model"),
            rev=current_test_branch,
        )

    return get


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "repo",
    ["simple", pytest.param("dvc_pipeline", marks=need_test_repo_ssh_auth)],
)
def test_remote_model_cloning(remote_model_meta, repo):
    with tempfile.TemporaryDirectory() as path:
        path = os.path.join(path, "model")
        remote_model_meta(repo).clone(path, link=False)
        cloned_model_meta = load_meta(path, load_value=False)
        _check_cloned_model(cloned_model_meta, path)


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "repo",
    ["simple", pytest.param("dvc_pipeline", marks=need_test_repo_ssh_auth)],
)
def test_remote_model_cloning_to_remote(
    remote_model_meta, repo, s3_tmp_path, s3_storage_fs
):
    path = s3_tmp_path("remote_model_cloning_to_remote")
    remote_model_meta(repo).clone(path, link=False)
    s3path = path[len("s3:/") :]
    assert s3_storage_fs.isfile(s3path + MLEM_EXT)
    assert s3_storage_fs.isfile(s3path)
    cloned_model_meta = load_meta(path, load_value=False)
    _check_cloned_model(cloned_model_meta, path, s3_storage_fs)


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
    assert os.path.abspath(
        mlem_dir_path(
            os.path.join(filled_mlem_repo, "data", "model"),
            obj_type="model",
            fs=None,
        )
    ) == os.path.abspath(model_link)
    # case when we provide object relative path
    model_link = os.path.join(
        filled_mlem_repo, MLEM_DIR, "model", "latest" + MLEM_EXT
    )
    assert os.path.abspath(
        mlem_dir_path(
            "latest", fs=None, obj_type="model", repo=filled_mlem_repo
        )
    ) == os.path.abspath(model_link)


def test_link_dump(model_path):
    link = MlemLink(
        path=model_path + MLEM_EXT,
        link_type="model",
    )
    with tempfile.TemporaryDirectory() as path:
        path_to_link = os.path.join(path, "latest" + MLEM_EXT)
        link.dump(path_to_link)
        model = load_meta(path_to_link, follow_links=True)
    assert isinstance(model, ModelMeta)


def test_double_link_load(filled_mlem_repo):
    latest = load_meta("latest", repo=filled_mlem_repo, follow_links=False)
    link = latest.make_link("external", repo=filled_mlem_repo, external=True)
    assert link.link_type == "model"
    model = load_meta("external", repo=filled_mlem_repo, follow_links=True)
    assert isinstance(model, ModelMeta)


@long
@need_test_repo_auth
def test_load_link_from_rev():
    # obj can be any mlem object, it's just easier to test with link
    obj = load(
        "rev_link/the_link",
        repo=MLEM_TEST_REPO,
        rev=REV_LINK_TAG,
        follow_links=False,
    )
    assert isinstance(obj, MlemLink)
    assert obj.path == "first"

    link = obj.make_link(absolute=True)
    link = link.deepcopy()
    loaded_link = link.load_link(follow_links=False)
    assert loaded_link == obj


def test_link_dump_in_mlem(model_path_mlem_repo):
    model_path, mlem_repo = model_path_mlem_repo
    link = MlemLink(
        path=model_path + MLEM_EXT,
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
    model = parse_obj_as(ModelMeta, payload)
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
    assert os.path.isfile(str(model_dir) + MLEM_EXT)
    assert os.path.isfile(str(model_dir))


class MockModelIO(ModelIO):
    filename: str

    def dump(self, storage: Storage, path, model) -> Artifacts:
        path = posixpath.join(path, self.filename)
        with storage.open(path) as (f, art):
            pickle.dump(model, f)
            return {self.filename: art}

    def load(self, artifacts: Artifacts):
        with artifacts[self.filename].open() as f:
            return pickle.load(f)


def test_remove_old_artifacts(model, tmpdir, train):
    model1 = ModelMeta.from_obj(model)
    model1.model_type.io = MockModelIO(filename="first.pkl")
    path = str(tmpdir / "model")
    model1.dump(path)
    assert os.path.isfile(tmpdir / "model.mlem")
    assert os.path.isdir(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model" / "first.pkl")
    load(path).predict(train)
    model2 = ModelMeta.from_obj(model)
    model2.model_type.io = MockModelIO(filename="second.pkl")
    model2.dump(path)
    assert os.path.isfile(tmpdir / "model.mlem")
    assert os.path.isdir(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model" / "second.pkl")
    assert not os.path.exists(tmpdir / "model" / "first.pkl")
    load(path).predict(train)
