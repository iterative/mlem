import os
import posixpath
import tempfile
from pathlib import Path
from typing import Any, Callable, Type

import git
import pandas as pd
import pytest
from fsspec.implementations.local import LocalFileSystem
from git import GitCommandError, Repo
from requests import ConnectionError, HTTPError
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem import CONFIG
from mlem.api import init, save
from mlem.constants import PREDICT_ARG_NAME, PREDICT_METHOD_NAME
from mlem.contrib.sklearn import SklearnModel
from mlem.core.artifacts import LOCAL_STORAGE, FSSpecStorage, LocalArtifact
from mlem.core.dataset_type import (
    Dataset,
    DatasetReader,
    DatasetType,
    DatasetWriter,
)
from mlem.core.meta_io import MLEM_EXT, get_fs
from mlem.core.metadata import load_meta
from mlem.core.model import Argument, ModelType, Signature
from mlem.core.objects import DatasetMeta, ModelMeta
from mlem.core.requirements import Requirements
from mlem.utils.github import ls_github_branches

RESOURCES = "resources"

long = pytest.mark.long
MLEM_TEST_REPO_ORG = "iterative"
MLEM_TEST_REPO_NAME = "mlem-test"
MLEM_TEST_REPO = (
    f"https://github.com/{MLEM_TEST_REPO_ORG}/{MLEM_TEST_REPO_NAME}/"
)

MLEM_S3_TEST_BUCKET = "mlem-tests"


def _check_github_test_repo_ssh_auth():
    try:
        git.cmd.Git().ls_remote(MLEM_TEST_REPO)
        return True
    except GitCommandError:
        return False


def _check_github_test_repo_auth():
    if not CONFIG.GITHUB_USERNAME or not CONFIG.GITHUB_TOKEN:
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


@pytest.fixture()
def current_test_branch():
    try:
        branch = Repo(str(Path(__file__).parent.parent)).active_branch.name
    except TypeError:
        # github actions/checkout leaves repo in detached head state
        # but it has env with branch name
        branch = os.environ.get("GITHUB_HEAD_REF", os.environ["GITHUB_REF"])
        if branch.startswith("refs/heads/"):
            branch = branch[len("refs/heads/") :]
    remote_refs = set(
        ls_github_branches(MLEM_TEST_REPO_ORG, MLEM_TEST_REPO_NAME).keys()
    )
    if branch in remote_refs:
        return branch
    return "main"


@pytest.fixture(scope="session", autouse=True)
def add_test_env():
    os.environ["MLEM_TESTS"] = "true"
    CONFIG.TESTS = True


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
        # TODO: https://github.com/iterative/mlem/issues/148
        train.columns = train.columns.astype(str)
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


@pytest.fixture
def dataset_meta(train):
    return DatasetMeta.from_data(train)


@pytest.fixture
def model_meta(model):
    return ModelMeta.from_obj(model)


@pytest.fixture
def model_path(model_train_target, tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "saved-model")
    model, train, _ = model_train_target
    # because of link=False we test reading by path here
    # reading by link name is not tested
    save(model, path, tmp_sample_data=train, link=False)
    yield path


@pytest.fixture
def data_path(train, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-data") / "data")
    save(train, temp_dir, link=False)
    yield temp_dir


@pytest.fixture
def dataset_meta_saved(data_path):
    return load_meta(data_path)


@pytest.fixture
def model_meta_saved(model_path):
    return load_meta(model_path)


@pytest.fixture
def model_meta_saved_single(tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "saved-model-single")
    train, target = load_iris(return_X_y=True)
    model = DecisionTreeClassifier().fit(train, target)
    return save(model, path, tmp_sample_data=train)


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
    model = ModelMeta(
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
def mlem_repo(tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("mlem-root"))
    init(dir)
    return dir


@pytest.fixture
def mlem_curdir_repo(tmpdir_factory):
    dir = str(tmpdir_factory.mktemp("mlem-root"))
    curdir = os.getcwd()
    os.chdir(dir)
    init()
    yield
    os.chdir(curdir)


@pytest.fixture
def filled_mlem_repo(mlem_repo):
    model = ModelMeta(
        requirements=Requirements.new("sklearn"),
        model_type=SklearnModel(methods={}, model=""),
    )
    model.dump("model1", repo=mlem_repo, external=True)

    model.make_link("latest", repo=mlem_repo)
    yield mlem_repo


@pytest.fixture
def model_path_mlem_repo(model_train_target, tmpdir_factory):
    model, train, _ = model_train_target
    dir = str(tmpdir_factory.mktemp("mlem-root-with-model"))
    init(dir)
    model_dir = os.path.join(dir, "generated-model")
    save(model, model_dir, tmp_sample_data=train, link=True, external=True)
    yield model_dir, dir


def dataset_write_read_check(
    dataset: Dataset,
    writer: DatasetWriter = None,
    reader_type: Type[DatasetReader] = None,
    custom_eq: Callable[[Any, Any], bool] = None,
    custom_assert: Callable[[Any, Any], Any] = None,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = writer or dataset.dataset_type.get_writer()

        storage = LOCAL_STORAGE
        reader, artifacts = writer.write(
            dataset, storage, posixpath.join(tmpdir, "data")
        )
        if reader_type is not None:
            assert isinstance(reader, reader_type)

        new = reader.read(artifacts)

        assert dataset.dataset_type == new.dataset_type
        if custom_assert is not None:
            custom_assert(new.data, dataset.data)
        else:
            if custom_eq is not None:
                assert custom_eq(new.data, dataset.data)
            else:
                assert new.data == dataset.data


def check_model_type_common_interface(
    model_type: ModelType,
    data_type: DatasetType,
    returns_type: DatasetType,
    **kwargs,
):
    assert PREDICT_METHOD_NAME in model_type.methods
    assert model_type.methods[PREDICT_METHOD_NAME] == Signature(
        name=PREDICT_METHOD_NAME,
        args=[Argument(name=PREDICT_ARG_NAME, type_=data_type)],
        returns=returns_type,
        **kwargs,
    )


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
def set_mlem_repo_root(mocker):
    def set(path, __file__=__file__):
        mocker.patch(
            "mlem.utils.root.find_repo_root",
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
