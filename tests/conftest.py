import os
import posixpath
import tempfile
from pathlib import Path
from typing import Any, Callable, Set, Type

import git
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from flaky import flaky  # noqa  # pylint: disable=unused-import
from fsspec.implementations.local import LocalFileSystem
from git import GitCommandError, Repo
from requests import ConnectionError, HTTPError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem import LOCAL_CONFIG
from mlem.api import init, save
from mlem.contrib.fastapi import FastAPIServer
from mlem.contrib.github import ls_github_branches
from mlem.contrib.sklearn import SklearnModel
from mlem.core.artifacts import LOCAL_STORAGE, FSSpecStorage, LocalArtifact
from mlem.core.data_type import DataReader, DataType, DataWriter
from mlem.core.meta_io import MLEM_EXT, get_fs
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemData, MlemModel
from mlem.core.requirements import Requirements
from mlem.runtime.interface import prepare_model_interface

RESOURCES = "resources"

long = pytest.mark.long
MLEM_TEST_REPO_ORG = "iterative"
MLEM_TEST_REPO_NAME = "mlem-test"
MLEM_TEST_REPO = (
    f"https://github.com/{MLEM_TEST_REPO_ORG}/{MLEM_TEST_REPO_NAME}/"
)

MLEM_S3_TEST_BUCKET = "mlem-tests"


def _cut_empty_lines(string):
    return "\n".join(line for line in string.splitlines() if line)


def _check_github_test_repo_ssh_auth():
    try:
        git.cmd.Git().ls_remote(MLEM_TEST_REPO)
        return True
    except GitCommandError:
        return False


def _check_github_test_repo_auth():
    if not LOCAL_CONFIG.GITHUB_USERNAME or not LOCAL_CONFIG.GITHUB_TOKEN:
        return False
    try:
        get_fs(MLEM_TEST_REPO)
        return True
    except (HTTPError, ConnectionError):
        return False


need_test_repo_auth = pytest.mark.skipif(
    not _check_github_test_repo_auth(),
    reason="No http credentials for remote repo",
)

need_test_repo_ssh_auth = pytest.mark.skipif(
    not _check_github_test_repo_ssh_auth(),
    reason="No ssh credentials for remote repo",
)


def get_current_test_branch(branch_list: Set[str]):
    try:
        branch = Repo(str(Path(__file__).parent.parent)).active_branch.name
    except TypeError:
        # github actions/checkout leaves repo in detached head state
        # but it has env with branch name
        branch = os.environ.get("GITHUB_HEAD_REF", os.environ["GITHUB_REF"])
        if branch.startswith("refs/heads/"):
            branch = branch[len("refs/heads/") :]
    if branch in branch_list:
        return branch
    return "main"


@pytest.fixture()
def current_test_branch():
    return get_current_test_branch(
        set(ls_github_branches(MLEM_TEST_REPO_ORG, MLEM_TEST_REPO_NAME).keys())
    )


@pytest.fixture(scope="session", autouse=True)
def add_test_env():
    os.environ["MLEM_TESTS"] = "true"
    LOCAL_CONFIG.TESTS = True


@pytest.fixture(scope="session", autouse=True)
def add_debug_env():
    os.environ["MLEM_DEBUG"] = "true"
    LOCAL_CONFIG.DEBUG = True


def resource_path(test_file, *paths):
    resources_dir = os.path.join(os.path.dirname(test_file), RESOURCES)
    return os.path.join(resources_dir, *paths)


@pytest.fixture
def local_fs():
    return LocalFileSystem()


@pytest.fixture(params=["numpy", "pandas"])
def model_train_target(request):
    """Note that in tests we often use both model and data,
    so having them compatible is a requirement for now.
    Though in future we may want to add tests with incompatible models and data
    """
    if request.param == "numpy":
        train, target = load_iris(return_X_y=True)
    elif request.param == "pandas":
        train, target = load_iris(return_X_y=True)
        train = pd.DataFrame(train)
    model = DecisionTreeClassifier().fit(train, target)
    return model, train, target


@pytest.fixture
def pandas_data():
    return pd.DataFrame([[1, 0], [0, 1]], columns=["a", "b"])


@pytest.fixture
def train(model_train_target):
    return model_train_target[1]


@pytest.fixture
def model(model_train_target):
    return model_train_target[0]


@pytest.fixture()
def server():
    return FastAPIServer(standardize=True)


@pytest.fixture
def interface(model, train, server):
    model = MlemModel.from_obj(model, sample_data=train)
    interface = prepare_model_interface(model, server)
    return interface


@pytest.fixture
def client(interface, server):
    app = server.app_init(interface)
    return TestClient(app)


@pytest.fixture
def request_get_mock(mocker, client):
    """
    mocks requests.get method so as to use
    FastAPI's TestClient's get() method beneath
    """

    def patched_get(url, params=None, **kwargs):
        url = url[len("http://") :]
        return client.get(url, params=params, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.requests.get",
        side_effect=patched_get,
    )


@pytest.fixture
def request_post_mock(mocker, client):
    """
    mocks requests.post method so as to use
    FastAPI's TestClient's post() method beneath
    """

    def patched_post(url, data=None, json=None, **kwargs):
        url = url[len("http://") :]
        return client.post(url, data=data, json=json, **kwargs)

    return mocker.patch(
        "mlem.runtime.client.requests.post",
        side_effect=patched_post,
    )


@pytest.fixture
def data_meta(train):
    return MlemData.from_data(train)


@pytest.fixture
def model_meta(model):
    return MlemModel.from_obj(model)


@pytest.fixture
def model_path(model_train_target, tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "saved-model")
    model, train, _ = model_train_target
    # because of index=False we test reading by path here
    # reading by link name is not tested
    save(model, path, sample_data=train)
    yield path


@pytest.fixture
def data_path(train, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-data") / "data")
    save(train, temp_dir)
    yield temp_dir


@pytest.fixture
def data_meta_saved(data_path):
    return load_meta(data_path)


@pytest.fixture
def model_meta_saved(model_path):
    return load_meta(model_path)


@pytest.fixture
def model_meta_saved_single(tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "saved-model-single")
    train, target = load_iris(return_X_y=True)
    model = DecisionTreeClassifier().fit(train, target)
    return save(model, path, sample_data=train)


@pytest.fixture
def model_single_path(model_meta_saved_single):
    return model_meta_saved_single.loc.uri


@pytest.fixture
def complex_model_meta_saved_single(tmp_path_factory):
    name = "saved-complex-model-single"
    path = os.path.join(tmp_path_factory.getbasetemp(), name)
    p = Path(path)
    p.mkdir(exist_ok=True)
    (p / "file1").write_text("data1", encoding="utf8")
    (p / "file2").write_text("data2", encoding="utf8")
    model = MlemModel(
        artifacts={
            "file1": LocalArtifact(
                uri=posixpath.join(name, "file1"), size=1, hash=""
            ),
            "file2": LocalArtifact(
                uri=posixpath.join(name, "file2"), size=1, hash=""
            ),
        },
        model_type=SklearnModel(methods={}),
    )
    return model.dump(path)


@pytest.fixture
def complex_model_single_path(complex_model_meta_saved_single):
    return complex_model_meta_saved_single.loc.uri


@pytest.fixture
def mlem_project(tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("mlem-root"))
    init(dir)
    return dir


@pytest.fixture
def mlem_curdir_project(tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("mlem-root"))
    curdir = os.getcwd()
    os.chdir(dir)
    init()
    yield
    os.chdir(curdir)


@pytest.fixture
def filled_mlem_project(mlem_project):
    model = MlemModel(
        requirements=Requirements.new("sklearn"),
        model_type=SklearnModel(methods={}, model=""),
    )
    model.dump("model1", project=mlem_project)

    model.make_link("latest", project=mlem_project)
    yield mlem_project


@pytest.fixture
def model_path_mlem_project(model_train_target, tmpdir_factory):
    model, train, _ = model_train_target
    dir = str(tmpdir_factory.mktemp("mlem-root-with-model"))
    init(dir)
    model_dir = os.path.join(dir, "generated-model")
    save(model, model_dir, sample_data=train)
    yield model_dir, dir


def data_write_read_check(
    data: DataType,
    writer: DataWriter = None,
    reader_type: Type[DataReader] = None,
    custom_eq: Callable[[Any, Any], bool] = None,
    custom_assert: Callable[[Any, Any], Any] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = writer or data.get_writer()

        storage = LOCAL_STORAGE
        reader, artifacts = writer.write(
            data, storage, posixpath.join(tmpdir, "data")
        )
        if reader_type is not None:
            assert isinstance(reader, reader_type)

        new = reader.read(artifacts)

        assert data == new
        if custom_assert is not None:
            custom_assert(new.data, data.data)
        else:
            if custom_eq is not None:
                assert custom_eq(new.data, data.data)
            else:
                assert new.data == data.data

        return artifacts


@pytest.fixture()
def github_matrix_os():
    return os.environ.get("GITHUB_MATRIX_OS", None)


@pytest.fixture()
def github_matrix_python():
    return os.environ.get("GITHUB_MATRIX_PYTHON", None)


@pytest.fixture
def s3_tmp_path(github_matrix_os, github_matrix_python):
    paths = set()
    base_path = f"s3://{MLEM_S3_TEST_BUCKET}"
    fs, _ = get_fs(base_path)

    def gen(path):
        path = posixpath.join(
            base_path, path + f"-{github_matrix_os}-{github_matrix_python}"
        )
        if path in paths:
            raise ValueError(f"Already generated {path}")
        if fs.exists(path):
            fs.delete(path, recursive=True)
        paths.add(path)
        return path

    yield gen
    for p in paths:
        try:
            fs.delete(p, recursive=True)
        except FileNotFoundError:
            pass
        try:
            fs.delete(p + MLEM_EXT)
        except FileNotFoundError:
            pass


@pytest.fixture()
def s3_storage():
    return FSSpecStorage(
        uri=f"s3://{MLEM_S3_TEST_BUCKET}/", storage_options={}
    )


@pytest.fixture()
def s3_storage_fs(s3_storage):
    return s3_storage.get_fs()


@pytest.fixture
def set_mlem_project_root(mocker):
    def set(path, __file__=__file__):
        mocker.patch(
            "mlem.utils.root.find_project_root",
            return_value=resource_path(__file__, path),
        )

    return set


def skip_matrix(os_: str, python: str):
    current_os = os.environ.get("GITHUB_MATRIX_OS")
    current_python = os.environ.get("GITHUB_MATRIX_PYTHON")
    if current_python is None or current_os is None:
        return lambda f: f
    if os_ == current_os and python == current_python:
        return lambda f: f
    return pytest.mark.skip(
        reason=f"This test is only for {os_} and python:{python}"
    )


@pytest.fixture(scope="session", autouse=True)
def disable_colorama():
    # workaround for tf+dvc import error
    # https://github.com/pytest-dev/pytest/issues/5502
    import colorama

    colorama.init = lambda: None


@pytest.fixture
def numpy_default_int_dtype():
    # default int type is platform dependent.
    # For windows 64 it is int32 and for linux 64 it is int64
    return str(np.array([1]).dtype)
