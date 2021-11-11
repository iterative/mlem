import os
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

import pytest
import yaml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlem.api import init
from mlem.constants import MLEM_DIR
from mlem.core.meta_io import ART_DIR, META_FILE_NAME
from mlem.core.metadata import load, load_meta, save
from mlem.core.objects import ModelMeta
from tests.conftest import (
    MLEM_TEST_REPO,
    MLEM_TEST_REPO_NAME,
    MLEM_TEST_REPO_ORG,
    long,
    need_test_repo_auth,
    need_test_repo_ssh_auth,
)


def test_model_saving_without_sample_data(model, tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("saving-models-without-sample-data"))
    # link=True would require having .mlem folder somewhere
    save(model, dir, link=False)


def test_model_saving_in_mlem_root(model_train_target, tmpdir_factory):
    mlem_root = str(tmpdir_factory.mktemp("mlem-root"))
    init(mlem_root)
    model_dir = os.path.join(mlem_root, "generated-model")
    model, train, _ = model_train_target
    save(model, model_dir, tmp_sample_data=train, link=True)


def test_model_saving(model_path):
    model_path = Path(model_path)
    assert os.path.isfile(model_path / META_FILE_NAME)
    assert os.path.isdir(model_path / ART_DIR)
    assert os.path.isfile(model_path / ART_DIR / "data.pkl")


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
    model = load_meta(model_path, load_value=True, force_type=ModelMeta)
    assert isinstance(model.model_type.model, DecisionTreeClassifier)
    train, _ = load_iris(return_X_y=True)
    model.model_type.model.predict(train)


@long
@need_test_repo_auth
@pytest.mark.parametrize(
    "url",
    [
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/data/model",
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/data/model/mlem.yaml",
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/.mlem/model/data/model.mlem.yaml",
        f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{{branch}}/simple/.mlem/model/latest.mlem.yaml",
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
        "simple/data/model",
        "simple/data/model/mlem.yaml",
        "simple/.mlem/model/data/model.mlem.yaml",
        "simple/.mlem/model/latest.mlem.yaml",
    ],
)
def test_model_loading_from_github(path, current_test_branch):
    assert "GITHUB_USERNAME" in os.environ and "GITHUB_TOKEN" in os.environ
    model = load(
        path,
        repo=MLEM_TEST_REPO,
        rev=current_test_branch,
    )
    train, _ = load_iris(return_X_y=True)
    model.predict(train)


@need_test_repo_auth
def test_load_link_with_fsspec_path(current_test_branch):
    link_contents = {
        "link_type": "model",
        "mlem_link": f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@{quote_plus(current_test_branch)}/simple/data/model/mlem.yaml",
        "object_type": "link",
    }
    with tempfile.TemporaryDirectory() as dir:
        path = os.path.join(dir, "link.mlem.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml.safe_dump(link_contents))
        model = load(path)
        train, _ = load_iris(return_X_y=True)
        model.predict(train)


@pytest.mark.xfail  # TODO: https://github.com/iterative/mlem/issues/110
@long
def test_saving_to_s3(model, s3_storage_fs, s3_tmp_path):
    path = s3_tmp_path("model_save")
    init(path)
    model_path = os.path.join(path, "model")
    save(model, model_path, fs=s3_storage_fs)
    model_path = Path(model_path[len("s3:/") :])
    assert s3_storage_fs.isfile(
        os.path.join(path, MLEM_DIR, "model", "model.mlem.yaml")
    )
    assert s3_storage_fs.isfile(str(model_path / META_FILE_NAME))
    assert s3_storage_fs.isdir(str(model_path / ART_DIR))
    assert s3_storage_fs.isfile(str(model_path / ART_DIR / "data.pkl"))


@pytest.mark.xfail  # TODO: https://github.com/iterative/mlem/issues/110
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
