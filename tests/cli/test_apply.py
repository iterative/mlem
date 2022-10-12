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
from mlem.core.errors import UnsupportedDataBatchLoading
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemData
from mlem.runtime.client import HTTPClient
from tests.cli.conftest import Runner
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
            ],
        )
        assert result.exit_code == 0, (
            result.stdout,
            result.stderr,
            result.exception,
        )
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
    save(model, path, sample_data=train)
    yield path


@pytest.fixture
def data_path_batch(model_train_batch, tmpdir_factory):
    temp_dir = str(tmpdir_factory.mktemp("saved-data") / "data")
    save(model_train_batch[1], temp_dir)
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
                "-b",
                "5",
            ],
        )
        assert result.exit_code == 0, (
            result.stdout,
            result.stderr,
            result.exception,
        )
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
                "--import",
                "--it",
                "pandas[csv]",
            ],
        )
        assert result.exit_code == 0, (
            result.stdout,
            result.stderr,
            result.exception,
        )
        predictions = load(path)
        assert isinstance(predictions, ndarray)


def test_apply_batch_with_import(
    runner: Runner, model_meta_saved_single, tmp_path_factory
):
    data_path = os.path.join(tmp_path_factory.getbasetemp(), "import_data")
    load_iris(return_X_y=True, as_frame=True)[0].to_csv(data_path, index=False)

    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        with pytest.raises(
            UnsupportedDataBatchLoading,
            match="Batch data loading is currently not supported for loading data on-the-fly",
        ):
            runner.invoke(
                [
                    "apply",
                    model_meta_saved_single.loc.uri,
                    data_path,
                    "-m",
                    "predict",
                    "-o",
                    path,
                    "--import",
                    "--it",
                    "pandas[csv]",
                    "-b",
                    "2",
                ],
                raise_on_error=True,
            )


def test_apply_no_output(runner, model_path, data_path):
    result = runner.invoke(
        ["apply", model_path, data_path, "-m", "predict"],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    assert len(result.stdout) > 0


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
        ],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    predictions = load(out)
    assert isinstance(predictions, ndarray)


def test_apply_remote(mlem_client, runner, data_path):
    with tempfile.TemporaryDirectory() as dir:
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            [
                "apply-remote",
                "http",
                "-d",
                data_path,
                "--host",
                "",
                "--port",
                "None",
                "-o",
                path,
            ],
            raise_on_error=True,
        )
        assert result.exit_code == 0, (
            result.stdout,
            result.stderr,
            result.exception,
        )
        predictions = load(path)
        assert isinstance(predictions, ndarray)
