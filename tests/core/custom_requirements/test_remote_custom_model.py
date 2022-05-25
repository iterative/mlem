import posixpath

from mlem.core.metadata import load_meta
from tests.conftest import MLEM_TEST_REPO, long, need_test_repo_auth


@long
@need_test_repo_auth
def test_remote_custom_model(current_test_branch):
    model_meta = load_meta(
        "custom_model",
        project=posixpath.join(MLEM_TEST_REPO, "custom_model"),
        rev=current_test_branch,
    )
    model_meta.load_value()
    model = model_meta.get_value()
    assert model.predict("b") == "ba"
