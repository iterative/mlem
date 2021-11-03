import os

import pytest
from click.testing import CliRunner

from mlem.cli import ls, pretty_print
from mlem.core.meta_io import META_FILE_NAME, get_path_by_repo_path_rev
from tests.conftest import MLEM_TEST_REPO, long

LOCAL_LS_EXPECTED_RESULT = """Models:
 - latest -> model1
 - model1
"""


@pytest.mark.parametrize("obj_type", [None, "all", "model"])
def test_ls(mlem_root, obj_type):
    runner = CliRunner()
    os.chdir(mlem_root)
    result = runner.invoke(
        ls,
        [obj_type] if obj_type else [],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0, "Output is empty, but should not be"
    assert result.output == LOCAL_LS_EXPECTED_RESULT


REMOTE_LS_EXPECTED_RESULT = """Models:
 - data/model
 - latest -> data/model
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
        [
            "all",
            "-r",
            f"{MLEM_TEST_REPO}/tree/main/simple",
        ],
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


@long
def test_pretty_print_remote(current_test_branch):
    model_path, _ = get_path_by_repo_path_rev(
        MLEM_TEST_REPO, "simple/data/model", current_test_branch
    )
    runner = CliRunner()
    result = runner.invoke(
        pretty_print,
        [os.path.join(model_path, META_FILE_NAME)],
    )
    assert result.exit_code == 0, (result.output, result.exception)
