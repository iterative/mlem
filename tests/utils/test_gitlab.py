import pytest

from mlem.core.errors import RevisionNotFound
from mlem.core.meta_io import UriResolver, get_fs
from mlem.utils.gitlabfs import GitlabFileSystem
from tests.conftest import long

MLEM_TEST_REPO_PROJECT = "mike0sv/fsspec-test"

MLEM_TEST_REPO_URI = f"https://gitlab.com/{MLEM_TEST_REPO_PROJECT}"


@long
def test_ls():
    fs = GitlabFileSystem(MLEM_TEST_REPO_PROJECT)
    assert fs.ls("") == ["README.md"]


@long
def test_open():
    fs = GitlabFileSystem(MLEM_TEST_REPO_PROJECT)
    with fs.open("README.md", "r") as f:
        assert f.read().startswith("# fsspec-test")


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
    ["main", "branch", "tag", "eb2ffd48b624589ce8e39b51a4b984f6887deeb5"],
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
