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
from mlem.core.errors import MlemProjectNotFound, WrongRequirementsError
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.metadata import load, load_meta
from mlem.core.model import ModelIO
from mlem.core.objects import (
    DeployState,
    MlemDeployment,
    MlemLink,
    MlemModel,
    MlemObject,
)
from mlem.core.requirements import InstallableRequirement, Requirements
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

    def get_client(self):
        pass


@pytest.fixture()
def meta():
    return MlemDeployment(
        env_link=MlemLink(path="", link_type="env"),
        model_link=MlemLink(path="", link_type="model"),
        state=MyDeployState(),
    )


@pytest.fixture(params=["fullpath", "with_root"])
def path_and_root(mlem_project, request):
    def get(name):
        if request.param == "fullpath":
            return os.path.join(mlem_project, name), None
        return name, mlem_project

    return get


@pytest.mark.parametrize("external", [True, False])
def test_meta_dump_curdir(meta, mlem_curdir_project, external):
    meta.dump(DEPLOY_NAME, external=external)
    path = DEPLOY_NAME + MLEM_EXT
    if not external:
        path = os.path.join(MLEM_DIR, meta.object_type, path)
    assert os.path.isfile(path)
    assert isinstance(load(DEPLOY_NAME), MlemDeployment)


def test_meta_dump__no_root(meta, tmpdir):
    with pytest.raises(MlemProjectNotFound):
        meta.dump(DEPLOY_NAME, project=str(tmpdir))


def test_meta_dump_fullpath_in_project_no_link(mlem_project, meta):
    meta.dump(
        os.path.join(mlem_project, MLEM_DIR, meta.object_type, DEPLOY_NAME),
        index=True,
        external=True,
    )
    link_path = os.path.join(
        mlem_project, MLEM_DIR, MlemLink.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert not os.path.exists(link_path)


def test_meta_dump_internal(mlem_project, meta, path_and_root):
    path, root = path_and_root(DEPLOY_NAME)
    meta.dump(path, project=root, external=False)
    assert meta.name == DEPLOY_NAME
    meta_path = os.path.join(
        mlem_project,
        MLEM_DIR,
        MlemDeployment.object_type,
        DEPLOY_NAME + MLEM_EXT,
    )
    assert os.path.isfile(meta_path)
    load_path = load_meta(meta_path)
    assert isinstance(load_path, MlemDeployment)
    assert load_path.name == meta.name
    load_root = load_meta(path, project=root)
    assert isinstance(load_root, MlemDeployment)
    assert load_root.name == meta.name


def test_meta_dump_external(mlem_project, meta, path_and_root):
    path, root = path_and_root(DEPLOY_NAME)
    meta.dump(path, project=root, external=True)
    assert meta.name == DEPLOY_NAME
    meta_path = os.path.join(mlem_project, DEPLOY_NAME + MLEM_EXT)
    assert os.path.isfile(meta_path)
    loaded = load_meta(meta_path)
    assert isinstance(loaded, MlemDeployment)
    assert loaded.name == meta.name
    link_path = os.path.join(
        mlem_project, MLEM_DIR, MlemLink.object_type, DEPLOY_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    assert isinstance(load_meta(link_path, follow_links=False), MlemLink)


@pytest.mark.parametrize("external", [False, True])
def test_model_dump_curdir(model_meta, mlem_curdir_project, external):
    model_meta.dump(MODEL_NAME, external=external)
    assert model_meta.name == MODEL_NAME
    if not external:
        prefix = Path(os.path.join(MLEM_DIR, model_meta.object_type))
    else:
        prefix = Path("")
    assert os.path.isfile(prefix / MODEL_NAME)
    assert os.path.isfile(prefix / (MODEL_NAME + MLEM_EXT))
    assert isinstance(load_meta(MODEL_NAME), MlemModel)


def test_model_dump_internal(mlem_project, model_meta, path_and_root):
    path, root = path_and_root(MODEL_NAME)
    model_meta.dump(path, project=root, external=False)
    assert model_meta.name == MODEL_NAME
    model_path = os.path.join(
        mlem_project, MLEM_DIR, MlemModel.object_type, MODEL_NAME
    )
    assert os.path.isfile(model_path + MLEM_EXT)
    assert os.path.isfile(model_path)


def test_model_dump_external(mlem_project, model_meta, path_and_root):
    path, root = path_and_root(MODEL_NAME)
    model_meta.dump(path, project=root, external=True)
    assert model_meta.name == MODEL_NAME
    model_path = os.path.join(mlem_project, MODEL_NAME)
    assert os.path.isfile(model_path + MLEM_EXT)
    assert os.path.isfile(model_path)
    link_path = os.path.join(
        mlem_project, MLEM_DIR, MlemLink.object_type, MODEL_NAME + MLEM_EXT
    )
    assert os.path.isfile(link_path)
    link = load_meta(link_path, follow_links=False)
    assert isinstance(link, MlemLink)
    model = link.load_link()
    assert model.dict() == model_meta.dict()


def _check_cloned_model(cloned_model_meta: MlemObject, path, fs=None):
    if fs is None:
        fs = LocalFileSystem()
    assert isinstance(cloned_model_meta, MlemModel)
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


def _check_complex_cloned_model(cloned_model_meta: MlemObject, path, fs=None):
    if fs is None:
        fs = LocalFileSystem()
    assert isinstance(cloned_model_meta, MlemModel)
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
        model.clone(path, index=False)
        cloned_model_meta = load_meta(path, load_value=False)
        _check_cloned_model(cloned_model_meta, path)


def test_complex_model_cloning(complex_model_single_path):
    model = load_meta(complex_model_single_path)
    with tempfile.TemporaryDirectory() as path:
        path = posixpath.join(path, "cloned")
        model.clone(path, index=False)
        cloned_model_meta = load_meta(path, load_value=False)
        _check_complex_cloned_model(cloned_model_meta, path)


@pytest.mark.parametrize("external", [True, False])
def test_model_cloning_to_project(model_single_path, mlem_project, external):
    model = load_meta(model_single_path)
    model.clone("model", project=mlem_project, index=False, external=external)
    cloned_model_meta = load_meta(
        "model", project=mlem_project, load_value=False
    )
    if external:
        path = os.path.join(mlem_project, "model")
    else:
        path = os.path.join(
            mlem_project, MLEM_DIR, MlemModel.object_type, "model"
        )
    _check_cloned_model(cloned_model_meta, path)


@long
def test_model_cloning_to_remote(model_path, s3_tmp_path, s3_storage_fs):
    model = load_meta(model_path)
    path = s3_tmp_path("model_cloning_to_remote")
    model.clone(path, index=False)
    s3path = path[len("s3:/") :]
    assert s3_storage_fs.isfile(s3path + MLEM_EXT)
    assert s3_storage_fs.isfile(s3path)
    cloned_model_meta = load_meta(path, load_value=False)
    _check_cloned_model(cloned_model_meta, path, s3_storage_fs)


@pytest.fixture
def remote_model_meta(current_test_branch):
    def get(project="simple"):
        return load_meta(
            os.path.join(MLEM_TEST_REPO, project, "data", "model"),
            rev=current_test_branch,
        )

    return get


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "project",
    ["simple", pytest.param("dvc_pipeline", marks=need_test_repo_ssh_auth)],
)
def test_remote_model_cloning(remote_model_meta, project):
    with tempfile.TemporaryDirectory() as path:
        path = os.path.join(path, "model")
        remote_model_meta(project).clone(path, index=False)
        cloned_model_meta = load_meta(path, load_value=False)
        _check_cloned_model(cloned_model_meta, path)


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "project",
    ["simple", pytest.param("dvc_pipeline", marks=need_test_repo_ssh_auth)],
)
def test_remote_model_cloning_to_remote(
    remote_model_meta, project, s3_tmp_path, s3_storage_fs
):
    path = s3_tmp_path("remote_model_cloning_to_remote")
    remote_model_meta(project).clone(path, index=False)
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


def test_link_dump(model_path):
    link = MlemLink(
        path=model_path + MLEM_EXT,
        link_type="model",
    )
    with tempfile.TemporaryDirectory() as path:
        path_to_link = os.path.join(path, "latest" + MLEM_EXT)
        link.dump(path_to_link)
        model = load_meta(path_to_link, follow_links=True)
    assert isinstance(model, MlemModel)


def test_double_link_load(filled_mlem_project):
    latest = load_meta(
        "latest", project=filled_mlem_project, follow_links=False
    )
    link = latest.make_link(
        "external", project=filled_mlem_project, external=True
    )
    assert link.link_type == "model"
    model = load_meta(
        "external", project=filled_mlem_project, follow_links=True
    )
    assert isinstance(model, MlemModel)


@long
@need_test_repo_auth
def test_load_link_from_rev():
    # obj can be any mlem object, it's just easier to test with link
    obj = load(
        "rev_link/the_link",
        project=MLEM_TEST_REPO,
        rev=REV_LINK_TAG,
        follow_links=False,
    )
    assert isinstance(obj, MlemLink)
    assert obj.path == "first"

    link = obj.make_link(absolute=True)
    link = link.deepcopy()
    loaded_link = link.load_link(follow_links=False)
    assert loaded_link == obj


def test_link_dump_in_mlem(model_path_mlem_project):
    model_path, mlem_project = model_path_mlem_project
    link = MlemLink(
        path=model_path + MLEM_EXT,
        link_type="model",
    )
    link_name = "latest"
    link.dump(link_name, project=mlem_project, external=True, index=False)
    model = load_meta(os.path.join(mlem_project, link_name), follow_links=True)
    assert isinstance(model, MlemModel)


def test_model_model_type_laziness():
    payload = {
        "model_type": {"type": "doesnotexist"},
        "object_type": "model",
        "requirements": [],
    }
    model = parse_obj_as(MlemModel, payload)
    assert model.model_type_raw == {"type": "doesnotexist"}
    with pytest.raises(ValidationError):
        print(model.model_type)


def test_mlem_project_root(filled_mlem_project):
    path = Path(filled_mlem_project)
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
    model1 = MlemModel.from_obj(model)
    model1.model_type.io = MockModelIO(filename="first.pkl")
    path = str(tmpdir / "model")
    model1.dump(path)
    assert os.path.isfile(tmpdir / "model.mlem")
    assert os.path.isdir(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model" / "first.pkl")
    load(path).predict(train)
    model2 = MlemModel.from_obj(model)
    model2.model_type.io = MockModelIO(filename="second.pkl")
    model2.dump(path)
    assert os.path.isfile(tmpdir / "model.mlem")
    assert os.path.isdir(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model" / "second.pkl")
    assert not os.path.exists(tmpdir / "model" / "first.pkl")
    load(path).predict(train)


def test_checkenv():
    model = MlemModel(
        requirements=Requirements.new(
            InstallableRequirement(module="pytest", version=pytest.__version__)
        )
    )

    model.checkenv()

    model.requirements.__root__[0].version = "asdasd"

    with pytest.raises(WrongRequirementsError):
        model.checkenv()
