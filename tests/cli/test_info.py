import json
import os

import pytest
from pydantic import parse_obj_as

from mlem.core.meta_io import MLEM_EXT
from mlem.core.objects import MlemLink, MlemMeta, ModelMeta
from tests.conftest import MLEM_TEST_REPO, long

LOCAL_LS_EXPECTED_RESULT = """Models:
 - latest -> model1
 - model1
"""


@pytest.mark.parametrize("obj_type", [None, "all", "model"])
def test_ls(runner, filled_mlem_repo, obj_type):
    os.chdir(filled_mlem_repo)
    result = runner.invoke(
        ["list", "-t", obj_type] if obj_type else ["list"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0, "Output is empty, but should not be"
    assert result.output == LOCAL_LS_EXPECTED_RESULT

    result = runner.invoke(
        (["list", "-t", obj_type] if obj_type else ["list"]) + ["--json"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    assert len(result.output) > 0, "Output is empty, but should not be"
    data = json.loads(result.output)
    assert "model" in data
    models = data["model"]
    assert len(models) == 2
    model, link = [parse_obj_as(MlemMeta, m) for m in models]
    if isinstance(model, MlemLink):
        model, link = link, model
    assert isinstance(model, ModelMeta)
    assert isinstance(link, MlemLink)


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

    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT, "--json"],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    meta = parse_obj_as(MlemMeta, json.loads(result.output))
    assert isinstance(meta, ModelMeta)


@long
def test_pretty_print_remote(runner, current_test_branch):
    model_path = os.path.join(
        MLEM_TEST_REPO, "tree", current_test_branch, "simple/data/model"
    )
    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (result.output, result.exception)
