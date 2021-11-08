import tempfile

from click.testing import CliRunner
from numpy import ndarray

from mlem.api.commands import load
from mlem.cli import apply
from mlem.core.meta_io import get_path_by_repo_path_rev
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth


def test_apply(model_path, data_path):
    with tempfile.TemporaryDirectory() as dir:
        runner = CliRunner()
        result = runner.invoke(
            apply,
            [model_path, data_path, "-m", "predict", "-o", dir, "--no-link"],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        predictions = load(dir)
        assert isinstance(predictions, ndarray)


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
        # TODO: https://github.com/iterative/mlem/issues/44
        #  add specific check for Exception/text in Exception


@long
@need_test_repo_auth
def test_apply_remote(current_test_branch, s3_tmp_path):
    runner = CliRunner()
    model_path, _ = get_path_by_repo_path_rev(
        MLEM_TEST_REPO, "simple/data/model", current_test_branch
    )
    data_path, _ = get_path_by_repo_path_rev(
        MLEM_TEST_REPO, "simple/data/test_x", current_test_branch
    )
    out = s3_tmp_path("apply_remote")
    result = runner.invoke(
        apply,
        [model_path, data_path, "-m", "predict", "-o", out, "--no-link"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    predictions = load(out)
    assert isinstance(predictions, ndarray)
