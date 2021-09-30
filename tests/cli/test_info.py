import os

import pytest
from click.testing import CliRunner

from mlem.cli import ls, pretty_print
from mlem.core.meta_io import META_FILE_NAME


@pytest.mark.parametrize("obj_type", [None, "all", "model", "dataset"])
def test_ls(mlem_root, obj_type):
    runner = CliRunner()
    os.chdir(mlem_root)
    result = runner.invoke(
        ls,
        [obj_type] if obj_type else [],
    )
    assert result.exit_code == 0, (result.output, result.exception)


def test_pretty_print(model_path_mlem_root):
    model_path, mlem_root = model_path_mlem_root
    runner = CliRunner()
    result = runner.invoke(
        pretty_print,
        [os.path.join(model_path, META_FILE_NAME)],
    )
    assert result.exit_code == 0, (result.output, result.exception)
