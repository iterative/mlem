import pytest

from mlem.contrib.gitlabfs import GitlabFileSystem, ls_gitlab_refs
from mlem.core.errors import RevisionNotFound
from mlem.core.meta_io import Location, get_fs
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemModel
from tests.conftest import get_current_test_branch, long

MLEM_TEST_REPO_PROJECT = "iterative.ai/mlem-test"

MLEM_TEST_REPO_URI = f"https://gitlab.com/{MLEM_TEST_REPO_PROJECT}"


@pytest.fixture()
def current_test_branch_gl():
    return get_current_test_branch(set(ls_gitlab_refs(MLEM_TEST_REPO_PROJECT)))


@long
def test_ls():
    fs = GitlabFileSystem(MLEM_TEST_REPO_PROJECT)
    assert "README.md" in fs.ls("")


@long
def test_open():
    fs = GitlabFileSystem(MLEM_TEST_REPO_PROJECT)
    with fs.open("README.md", "r") as f:
        assert f.read().startswith("### Fixture for mlem tests")


@long
@pytest.mark.parametrize(
    "uri",
    [
        MLEM_TEST_REPO_URI + "/-/blob/main/path",
        f"gitlab://{MLEM_TEST_REPO_PROJECT}@main/path",
    ],
)
def test_uri_resolver(uri):
    fs, path = get_fs(uri)

    assert isinstance(fs, GitlabFileSystem)
    assert path == "path"


@long
@pytest.mark.parametrize(
    "rev",
    ["main", "branch", "tag", "3897d2ab"],
)
def test_uri_resolver_rev(rev):
    location = Location.resolve(MLEM_TEST_REPO_URI, None, rev=rev, fs=None)
    assert isinstance(location.fs, GitlabFileSystem)
    assert location.fs.root == rev
    assert "README.md" in location.fs.ls("")


@long
def test_uri_resolver_wrong_rev():
    with pytest.raises(RevisionNotFound):
        Location.resolve(
            MLEM_TEST_REPO_URI, None, rev="__not_exists__", fs=None
        )


@long
def test_loading_object(current_test_branch_gl):
    meta = load_meta(
        "latest",
        project=MLEM_TEST_REPO_URI + "/-/blob/main/simple",
        rev=current_test_branch_gl,
    )
    assert isinstance(meta, MlemModel)
