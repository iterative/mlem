import os

import pytest

from mlem.core.meta_io import MLEM_EXT
from tests.conftest import MLEM_TEST_REPO, long

LOCAL_LS_EXPECTED_RESULT = """Models:
 - latest -> model1
 - model1
"""


@pytest.mark.parametrize("obj_type", [None, "all", "model"])
def test_ls(runner, filled_mlem_repo, obj_type):
    os.chdir(filled_mlem_repo)
    result = runner.invoke(
        ["list", obj_type] if obj_type else ["list"],
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
def test_ls_remote(runner, current_test_branch):
    result = runner.invoke(
        [
            "list",
            "all",
            "-r",
            f"{MLEM_TEST_REPO}/tree/{current_test_branch}/simple",
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0, "Output is empty, but should not be"
    assert result.output == REMOTE_LS_EXPECTED_RESULT


def test_pretty_print(runner, model_path_mlem_repo):
    model_path, _ = model_path_mlem_repo
    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (result.output, result.exception)


@long
def test_pretty_print_remote(runner, current_test_branch):
    model_path = os.path.join(
        MLEM_TEST_REPO, "tree", current_test_branch, "simple/data/model"
    )
    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (result.output, result.exception)
