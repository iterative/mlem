import pytest

from mlem.contrib.gitlabfs import GitlabFileSystem
from mlem.core.errors import RevisionNotFound
from mlem.core.meta_io import UriResolver, get_fs
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemModel
from tests.conftest import long

MLEM_TEST_REPO_PROJECT = "iterative.ai/mlem-test"

MLEM_TEST_REPO_URI = f"https://gitlab.com/{MLEM_TEST_REPO_PROJECT}"


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
    location = UriResolver.resolve(MLEM_TEST_REPO_URI, None, rev=rev, fs=None)
    assert isinstance(location.fs, GitlabFileSystem)
    assert location.fs.root == rev
    assert "README.md" in location.fs.ls("")


@long
def test_uri_resolver_wrong_rev():
    with pytest.raises(RevisionNotFound):
        UriResolver.resolve(
            MLEM_TEST_REPO_URI, None, rev="__not_exists__", fs=None
        )


@long
def test_loading_object():
    meta = load_meta(
        "latest", project=MLEM_TEST_REPO_URI + "/-/blob/main/simple"
    )
    assert isinstance(meta, MlemModel)
