import os
import tempfile

from mlem.api import load_meta
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT
from mlem.core.objects import MlemLink, ModelMeta


def test_link(runner, model_path):
    with tempfile.TemporaryDirectory() as dir:
        link_path = os.path.join(dir, "latest.mlem")
        result = runner.invoke(
            ["link", model_path, link_path, "-e", "--abs"],
        )
        assert result.exit_code == 0, (result.output, result.exception)
        assert os.path.exists(link_path)
        model = load_meta(link_path)
        assert isinstance(model, ModelMeta)


def test_link_mlem_repo(runner, model_path_mlem_repo):
    model_path, repo = model_path_mlem_repo
    link_name = "latest.mlem"
    result = runner.invoke(
        ["link", model_path, link_name, "--target-repo", repo],
    )
    assert result.exit_code == 0, result.output
    link_path = os.path.join(repo, MLEM_DIR, MlemLink.object_type, link_name)
    assert os.path.exists(link_path)
    link_object = load_meta(link_path, follow_links=False)
    assert isinstance(link_object, MlemLink)
    assert link_object.path[: -len(MLEM_EXT)] == os.path.basename(model_path)
    model = load_meta(link_path)
    assert isinstance(model, ModelMeta)
