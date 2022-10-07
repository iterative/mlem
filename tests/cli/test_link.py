import os
import tempfile

from mlem.api import load_meta
from mlem.core.meta_io import MLEM_EXT
from mlem.core.objects import MlemLink, MlemModel


def test_link(runner, model_path):
    with tempfile.TemporaryDirectory() as dir:
        link_path = os.path.join(dir, "latest.mlem")
        result = runner.invoke(
            ["link", model_path, link_path, "--abs"],
        )
        assert result.exit_code == 0, (
            result.stdout,
            result.stderr,
            result.exception,
        )
        assert os.path.exists(link_path)
        model = load_meta(link_path)
        assert isinstance(model, MlemModel)


def test_link_mlem_project(runner, model_path_mlem_project):
    model_path, project = model_path_mlem_project
    link_name = "latest.mlem"
    result = runner.invoke(
        ["link", model_path, link_name, "--target-project", project],
    )
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    link_path = os.path.join(project, link_name)
    assert os.path.exists(link_path)
    link_object = load_meta(link_path, follow_links=False)
    assert isinstance(link_object, MlemLink)
    assert link_object.path[: -len(MLEM_EXT)] == os.path.basename(model_path)
    model = load_meta(link_path)
    assert isinstance(model, MlemModel)
