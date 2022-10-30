import os
import posixpath
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

import pytest
import yaml
from pytest_lazyfixture import lazy_fixture
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlem.api import init
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import load, load_meta, save
from mlem.core.objects import MlemModel
from tests.conftest import (
    MLEM_TEST_REPO,
    MLEM_TEST_REPO_NAME,
    MLEM_TEST_REPO_ORG,
    long,
    need_test_repo_auth,
    need_test_repo_ssh_auth,
)


@pytest.mark.parametrize("obj", [lazy_fixture("model"), lazy_fixture("train")])
def test_save_with_meta_fields(obj, tmpdir):
    path = str(tmpdir / "obj")
    save(obj, path, params={"a": "b"})
    new = load_meta(path)
    assert new.params == {"a": "b"}


def test_saving_with_project(model, tmpdir):
    path = str(tmpdir / "obj")
    save(model, path)
    load_meta(path)


def test_saving_with_pathlib(model, tmpdir):
    # by default, tmpdir is of type: `py._path.local.LocalPath`,
    # see the test below
    path = Path(tmpdir) / "obj"
    save(model, path)
    load_meta(path)


def test_saving_with_localpath(model, tmpdir):
    path = tmpdir / "obj"
    save(model, path)
    load_meta(path)


def test_model_saving_without_sample_data(model, tmpdir_factory):
    path = str(
        tmpdir_factory.mktemp("saving-models-without-sample-data") / "model"
    )
    # index=True would require having .mlem folder somewhere
    save(model, path)


def test_model_saving_in_mlem_project_root(model_train_target, tmpdir_factory):
    project = str(tmpdir_factory.mktemp("mlem-root"))
    init(project)
    model_dir = os.path.join(project, "generated-model")
    model, train, _ = model_train_target
    save(model, model_dir, sample_data=train)


def test_model_saving(model_path):
    assert os.path.isfile(model_path + MLEM_EXT)
    assert os.path.isfile(model_path)


def test_model_loading(model_path):
    model = load(model_path)
    assert isinstance(model, DecisionTreeClassifier)
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


@long
@need_test_repo_ssh_auth
def test_model_loading_remote_dvc(current_test_branch):
    model = load(
        f"{MLEM_TEST_REPO}/dvc_pipeline/data/model",
        rev=current_test_branch,
    )
    assert isinstance(model, RandomForestClassifier)
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


def test_meta_loading(model_path):
    model = load_meta(model_path, load_value=True, force_type=MlemModel)
    assert isinstance(model.model_type.model, DecisionTreeClassifier)
    train, _ = load_iris(return_X_y=True)
    model.model_type.model.predict(train)


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "url",
    [
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/data/model",
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/data/model.mlem",
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/latest.mlem",
        f"{MLEM_TEST_REPO}tree/{{branch}}/simple/data/model/",
    ],
)
def test_model_loading_from_github_with_fsspec(url, current_test_branch):
    assert "GITHUB_USERNAME" in os.environ and "GITHUB_TOKEN" in os.environ
    model = load(url.format(branch=quote_plus(current_test_branch)))
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "path",
    [
        "data/model",
        "data/model.mlem",
        "latest.mlem",
    ],
)
def test_model_loading_from_github(path, current_test_branch):
    assert "GITHUB_USERNAME" in os.environ and "GITHUB_TOKEN" in os.environ
    model = load(
        path,
        project=os.path.join(MLEM_TEST_REPO, "simple"),
        rev=current_test_branch,
    )
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


@long
@need_test_repo_auth
def test_load_link_with_fsspec_path(current_test_branch):
    link_contents = {
        "link_type": "model",
        "path": f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{quote_plus(current_test_branch)}/simple/data/model.mlem",
        "object_type": "link",
    }
    with tempfile.TemporaryDirectory() as dirname:
        path = os.path.join(dirname, "link.mlem")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(link_contents))
        model = load(path)
        train, _ = load_iris(return_X_y=True)
        model.predict(train)


@long
def test_saving_to_s3(model, s3_storage_fs, s3_tmp_path):
    path = s3_tmp_path("model_save")
    init(path)
    model_path = posixpath.join(path, "model")
    save(model, model_path, fs=s3_storage_fs)
    model_path = model_path[len("s3:/") :]
    assert s3_storage_fs.isfile(posixpath.join(path, "model.mlem"))
    assert s3_storage_fs.isfile(model_path + MLEM_EXT)
    assert s3_storage_fs.isfile(model_path)


@long
def test_loading_from_s3(model, s3_storage_fs, s3_tmp_path):
    path = s3_tmp_path("model_load")
    init(path)
    model_path = os.path.join(path, "model")
    save(model, model_path, fs=s3_storage_fs)

    loaded = load(model_path)
    assert isinstance(loaded, DecisionTreeClassifier)
    train, _ = load_iris(return_X_y=True)
    loaded.predict(train)
