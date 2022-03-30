import os

import pytest
from typer.testing import CliRunner

from mlem.cli import app
from mlem.core.meta_io import MLEM_EXT
from tests.conftest import MLEM_TEST_REPO, long

LOCAL_LS_EXPECTED_RESULT = """Models:
 - latest -> model1
 - model1
"""


@pytest.mark.parametrize("obj_type", [None, "all", "model"])
def test_ls(filled_mlem_repo, obj_type):
    runner = CliRunner()
    os.chdir(filled_mlem_repo)
    result = runner.invoke(
        app,
        ["ls", obj_type] if obj_type else ["ls"],
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
def test_ls_remote(current_test_branch):
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ls",
            "all",
            "-r",
            f"{MLEM_TEST_REPO}/tree/{current_test_branch}/simple",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0, "Output is empty, but should not be"
    assert result.output == REMOTE_LS_EXPECTED_RESULT


def test_pretty_print(model_path_mlem_repo):
    model_path, _ = model_path_mlem_repo
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (result.output, result.exception)


@long
def test_pretty_print_remote(current_test_branch):
    model_path = os.path.join(
        MLEM_TEST_REPO, "tree", current_test_branch, "simple/data/model"
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (result.output, result.exception)
