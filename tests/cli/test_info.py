import json
import os

from pydantic import parse_obj_as

from mlem.core.meta_io import MLEM_EXT
from mlem.core.objects import MlemModel, MlemObject
from tests.conftest import MLEM_TEST_REPO, long


def test_pretty_print(runner, model_path_mlem_project):
    model_path, _ = model_path_mlem_project
    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )

    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT, "--json"],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    meta = parse_obj_as(MlemObject, json.loads(result.stdout))
    assert isinstance(meta, MlemModel)


@long
def test_pretty_print_remote(runner, current_test_branch):
    model_path = os.path.join(
        MLEM_TEST_REPO, "tree", current_test_branch, "simple/data/model"
    )
    result = runner.invoke(
        ["pprint", model_path + MLEM_EXT],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
