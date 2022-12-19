import os
import posixpath
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote_plus

import pytest
import yaml
from git import Repo
from pytest_lazyfixture import lazy_fixture
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from mlem.api import init
from mlem.contrib.heroku.meta import HerokuEnv
from mlem.core.errors import InvalidArgumentError
from mlem.core.meta_io import MLEM_EXT
from mlem.core.metadata import (
    list_objects,
    load,
    load_meta,
    log_meta_params,
    save,
)
from mlem.core.objects import MlemData, MlemEnv, MlemLink, MlemModel
from mlem.telemetry import pass_telemetry_params, telemetry
from mlem.utils.path import make_posix
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


def test_ls_local(filled_mlem_project):
    objects = list_objects(filled_mlem_project)

    assert len(objects) == 1
    assert MlemModel in objects
    models = objects[MlemModel]
    assert len(models) == 2
    model, lnk = models
    if isinstance(model, MlemLink):
        model, lnk = lnk, model

    assert isinstance(model, MlemModel)
    assert isinstance(lnk, MlemLink)
    assert (
        posixpath.join(make_posix(filled_mlem_project), lnk.path)
        == model.loc.fullpath
    )


@pytest.mark.parametrize("recursive,count", [[True, 3], [False, 1]])
def test_ls_local_recursive(tmpdir, recursive, count):
    path = str(tmpdir)
    meta = HerokuEnv()
    meta.dump(posixpath.join(path, "env"))
    meta.dump(posixpath.join(path, "subdir", "env"))
    meta.dump(posixpath.join(path, "subdir", "subsubdir", "env"))
    objects = list_objects(path, recursive=recursive)
    assert len(objects) == 1
    assert MlemEnv in objects
    assert len(objects[MlemEnv]) == count


def test_ls_no_project(tmpdir):
    assert not list_objects(str(tmpdir))


@long
@need_test_repo_auth
def test_ls_remote(current_test_branch):
    objects = list_objects(
        os.path.join(MLEM_TEST_REPO, f"tree/{current_test_branch}/simple")
    )
    assert len(objects) == 2
    assert MlemModel in objects
    models = objects[MlemModel]
    assert len(models) == 2
    model, lnk = models
    if isinstance(model, MlemLink):
        model, lnk = lnk, model

    assert isinstance(model, MlemModel)
    assert isinstance(lnk, MlemLink)

    assert MlemData in objects
    assert len(objects[MlemData]) == 4


@long
def test_ls_remote_s3(s3_tmp_path):
    path = s3_tmp_path("ls_remote_s3")
    init(path)
    meta = HerokuEnv()
    meta.dump(posixpath.join(path, "env"))
    meta.dump(posixpath.join(path, "subdir", "env"))
    meta.dump(posixpath.join(path, "subdir", "subsubdir", "env"))
    objects = list_objects(path)
    assert MlemEnv in objects
    envs = objects[MlemEnv]
    assert len(envs) == 3
    assert all(o == meta for o in envs)


@pytest.mark.xfail(
    sys.platform == "win32",
    reason="https://github.com/fsspec/filesystem_spec/issues/1125",
)
def test_load_local_rev(tmpdir):
    path = str(tmpdir / "obj")

    def model1(data):
        return data + 1

    def model2(data):
        return data + 2

    repo = Repo.init(str(tmpdir))
    save(model1, path)
    repo.index.add(["obj", "obj.mlem"])
    first = repo.index.commit("init")

    save(model2, path)
    repo.index.add(["obj", "obj.mlem"])
    second = repo.index.commit("second")

    assert load(path, rev=first.hexsha)(1) == 2
    assert load(path, rev=second.hexsha)(1) == 3
    assert load(path)(1) == 3

    shutil.rmtree(tmpdir / ".git")

    with pytest.raises(
        InvalidArgumentError,
        match=f"Rev `{first.hexsha}` was provided, but FSSpecResolver does not support versioning",
    ):
        load(path, rev=first.hexsha)


_data = MlemData(reader=None)
_data.reader_raw = {"data_type": {"type": "kek"}}


@pytest.mark.parametrize(
    "obj,params",
    [
        (MlemModel(), {}),
        (MlemModel(processors={"a": {"type": "kek"}}), {"model_type": "kek"}),
        (
            MlemModel(
                processors={"a": {"type": "kek"}, "b": {"type": "callable"}}
            ),
            {"model_type": "kek"},
        ),
        (
            MlemModel(processors={"a": {"type": "callable"}}),
            {"model_type": "callable"},
        ),
        (_data, {"data_type": "kek"}),
    ],
)
def test_log_meta_params(obj, params):
    with telemetry.event_scope(
        "test", "log_meta_params"
    ) as event, pass_telemetry_params():
        log_meta_params(obj)
        assert event.kwargs == params
