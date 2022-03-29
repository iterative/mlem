import os.path
import posixpath
import tempfile

from numpy import ndarray
from sklearn.datasets import load_iris
from typer.testing import CliRunner

from mlem.api import load
from mlem.cli import app
from mlem.core.errors import MlemRootNotFound
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth


def test_apply(model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        runner = CliRunner()
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            app,
            [
                "apply",
                model_path,
                data_path,
                "-m",
                "predict",
                "-o",
                path,
                "--no-link",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(path)
        assert isinstance(predictions, ndarray)


def test_apply_with_import(model_meta_saved_single, tmp_path_factory):
    data_path = os.path.join(tmp_path_factory.getbasetemp(), "import_data")
    load_iris(return_X_y=True, as_frame=True)[0].to_csv(data_path, index=False)

    with tempfile.TemporaryDirectory() as dir:
        runner = CliRunner()
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            app,
            [
                "apply",
                model_meta_saved_single.loc.uri,
                data_path,
                "-m",
                "predict",
                "-o",
                path,
                "--no-link",
                "--import",
                "--it",
                "pandas[csv]",
            ],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(path)
        assert isinstance(predictions, ndarray)


def test_apply_no_output(model_path, data_path):
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["apply", model_path, data_path, "-m", "predict", "--no-link"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0


def test_apply_fails_without_mlem_dir(model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "--tb",
                "apply",
                model_path,
                data_path,
                "-m",
                "predict",
                "-o",
                dir,
                "--link",
            ],
        )
        assert result.exit_code == 1, (result.output, result.exception)
        assert isinstance(result.exception, MlemRootNotFound)


@long
@need_test_repo_auth
def test_apply_from_remote(current_test_branch, s3_tmp_path):
    runner = CliRunner()
    model_path = "simple/data/model"
    data_path = "simple/data/test_x"
    out = s3_tmp_path("apply_remote")
    result = runner.invoke(
        app,
        [
            "apply",
            model_path,
            "--repo",
            MLEM_TEST_REPO,
            "--rev",
            current_test_branch,
            "-m",
            "predict",
            data_path,
            "--data-repo",
            MLEM_TEST_REPO,
            "--data-rev",
            current_test_branch,
            "-o",
            out,
            "--no-link",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    predictions = load(out)
    assert isinstance(predictions, ndarray)
