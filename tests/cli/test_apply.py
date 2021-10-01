import tempfile

from click.testing import CliRunner
from numpy import ndarray

from mlem.api.commands import load
from mlem.cli import apply


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
