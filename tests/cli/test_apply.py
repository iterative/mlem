import os.path
import posixpath
import tempfile

import pytest
from numpy import ndarray
from pandas import DataFrame
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from mlem.api import load, save
from mlem.core.data_type import ArrayType
from mlem.core.errors import MlemProjectNotFound
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemData
from mlem.runtime.client import HTTPClient
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth


@pytest.fixture
def mlem_client(request_get_mock, request_post_mock):
    client = HTTPClient(host="", port=None)
    return client


def test_apply(runner, model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            [
                "apply",
                model_path,
                data_path,
                "-m",
                "predict",
                "-o",
                path,
                "--no-index",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(path)
        assert isinstance(predictions, ndarray)


@pytest.fixture
def model_train_batch():
    train, target = load_iris(return_X_y=True)
    train = DataFrame(train)
    train.columns = train.columns.astype(str)
    model = DecisionTreeClassifier().fit(train, target)
    return model, train


@pytest.fixture
def model_path_batch(model_train_batch, tmp_path_factory):
    path = os.path.join(tmp_path_factory.getbasetemp(), "saved-model")
    model, train = model_train_batch
    save(model, path, sample_data=train, index=False)
    yield path


@pytest.fixture
def data_path_batch(model_train_batch, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-data") / "data")
    save(model_train_batch[1], temp_dir, index=False)
    yield temp_dir


def test_apply_batch(runner, model_path_batch, data_path_batch):
    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            [
                "--tb",
                "apply",
                model_path_batch,
                data_path_batch,
                "-m",
                "predict",
                "-o",
                path,
                "--no-index",
                "-b",
                "5",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions_meta = load_meta(
            path, load_value=True, force_type=MlemData
        )
        assert isinstance(predictions_meta.data_type, ArrayType)
        predictions = predictions_meta.get_value()
        assert isinstance(predictions, list)


def test_apply_with_import(runner, model_meta_saved_single, tmp_path_factory):
    data_path = os.path.join(tmp_path_factory.getbasetemp(), "import_data")
    load_iris(return_X_y=True, as_frame=True)[0].to_csv(data_path, index=False)

    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            [
                "--tb",
                "apply",
                model_meta_saved_single.loc.uri,
                data_path,
                "-m",
                "predict",
                "-o",
                path,
                "--no-index",
                "--import",
                "--it",
                "pandas[csv]",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(path)
        assert isinstance(predictions, ndarray)


def test_apply_batch_with_import(
    runner, model_meta_saved_single, tmp_path_factory
):
    data_path = os.path.join(tmp_path_factory.getbasetemp(), "import_data")
    load_iris(return_X_y=True, as_frame=True)[0].to_csv(data_path, index=False)

    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            [
                "apply",
                model_meta_saved_single.loc.uri,
                data_path,
                "-m",
                "predict",
                "-o",
                path,
                "--no-index",
                "--import",
                "--it",
                "pandas[csv]",
                "-b",
                "2",
            ],
        )
        assert result.exit_code == 1, (result.output, result.exception)
        assert (
            "Batch data loading is currently not supported for loading data on-the-fly"
            in result.output
        )


def test_apply_no_output(runner, model_path, data_path):
    result = runner.invoke(
        ["apply", model_path, data_path, "-m", "predict", "--no-index"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0


def test_apply_fails_without_mlem_dir(runner, model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        result = runner.invoke(
            [
                "--tb",
                "apply",
                model_path,
                data_path,
                "-m",
                "predict",
                "-o",
                dir,
                "--index",
            ],
        )
        assert result.exit_code == 1, (result.output, result.exception)
        assert isinstance(result.exception, MlemProjectNotFound)


@long
@need_test_repo_auth
def test_apply_from_remote(runner, current_test_branch, s3_tmp_path):
    model_path = "simple/data/model"
    data_path = "simple/data/test_x"
    out = s3_tmp_path("apply_remote")
    result = runner.invoke(
        [
            "apply",
            model_path,
            "--project",
            MLEM_TEST_REPO,
            "--rev",
            current_test_branch,
            "-m",
            "predict",
            data_path,
            "--data-project",
            MLEM_TEST_REPO,
            "--data-rev",
            current_test_branch,
            "-o",
            out,
            "--no-index",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    predictions = load(out)
    assert isinstance(predictions, ndarray)


def test_apply_remote(mlem_client, runner, data_path):
    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            [
                "apply-remote",
                "http",
                data_path,
                "-c",
                "host=''",
                "-c",
                "port=None",
                "-o",
                path,
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(path)
        assert isinstance(predictions, ndarray)
