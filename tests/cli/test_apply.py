import os.path
import posixpath
import tempfile

from click.testing import CliRunner
from numpy import ndarray

from mlem.api import load
from mlem.cli import apply
from mlem.core.errors import MlemRootNotFound
from tests.conftest import MLEM_TEST_REPO, issue_110, long, need_test_repo_auth


def test_apply(model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        runner = CliRunner()
        path = posixpath.join(dir, "data")
        result = runner.invoke(
            apply,
            [model_path, data_path, "-m", "predict", "-o", path, "--no-link"],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(path)
        assert isinstance(predictions, ndarray)


def test_apply_no_output(model_path, data_path):
    runner = CliRunner()
    result = runner.invoke(
        apply,
        [model_path, data_path, "-m", "predict", "--no-link"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    applying = "applying\n"
    assert result.output.startswith(applying)
    assert len(result.output) > len(applying)


def test_apply_for_multiple_datasets(model_path, data_path):
    runner = CliRunner()
    result = runner.invoke(
        apply,
        [model_path, data_path, data_path, "-m", "predict", "--no-link"],
    )
    assert result.exit_code == 0, (result.output, result.exception)


def test_apply_fails_without_mlem_dir(model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        runner = CliRunner()
        result = runner.invoke(
            apply,
            [model_path, data_path, "-m", "predict", "-o", dir, "--link"],
        )
        assert result.exit_code == 1, (result.output, result.exception)
        assert isinstance(result.exception, MlemRootNotFound)


@long
@need_test_repo_auth
@issue_110
def test_apply_remote(current_test_branch, s3_tmp_path):
    runner = CliRunner()
    model_path = os.path.join(
        MLEM_TEST_REPO, "tree", current_test_branch, "simple/data/model"
    )
    data_path = os.path.join(
        MLEM_TEST_REPO, "tree", current_test_branch, "simple/data/test_x"
    )
    out = s3_tmp_path("apply_remote")
    result = runner.invoke(
        apply,
        [model_path, data_path, "-m", "predict", "-o", out, "--no-link"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    predictions = load(out)
    assert isinstance(predictions, ndarray)
