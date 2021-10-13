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


REMOTE_LS_EXPECTED_RESULT = """Models:
 - data/model
 - latest -> data/model/mlem.yaml
Datasets:
 - data/test_x
 - data/test_y
 - data/train
"""


@pytest.mark.long
def test_ls_remote():
    runner = CliRunner()
    result = runner.invoke(
        ls,
        ["all", "-r", "https://github.com/iterative/example-mlem/"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0, "Output is empty, but should not be"
    assert result.output == REMOTE_LS_EXPECTED_RESULT


def test_pretty_print(model_path_mlem_root):
    model_path, _ = model_path_mlem_root
    runner = CliRunner()
    result = runner.invoke(
        pretty_print,
        [os.path.join(model_path, META_FILE_NAME)],
    )
    assert result.exit_code == 0, (result.output, result.exception)
