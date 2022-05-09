import pytest
from pytest_lazyfixture import lazy_fixture

from mlem.core.errors import RevisionNotFound
from mlem.core.meta_io import UriResolver, get_fs
from mlem.utils.bitbucketfs import CONFIG, BitBucketFileSystem
from tests.conftest import long

MLEM_TEST_REPO_PROJECT = "mike0sv/fsspec-test"

MLEM_TEST_REPO_URI = f"https://bitbucket.org/{MLEM_TEST_REPO_PROJECT}"


@pytest.fixture()
def fs_no_auth():
    username = CONFIG.USERNAME
    try:
        CONFIG.USERNAME = None
        yield BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)
    finally:
        CONFIG.USERNAME = username


@pytest.fixture()
def fs_auth():
    return BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)


@long
@pytest.mark.parametrize(
    "fs",
    [
        lazy_fixture("fs_auth"),
        # lazy_fixture("fs_no_auth")
    ],
)
def test_ls(fs):
    assert "README.md" in fs.ls("")


@long
def test_open():
    fs = BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)
    with fs.open("README.md", "r") as f:
        assert f.read().startswith("# README")


@long
def test_ls_auth():
    fs = BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)
    assert "README.md" in fs.ls("")


@long
def test_open_auth():
    fs = BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)
    with fs.open("README.md", "r") as f:
        assert f.read().startswith("# README")


@long
@pytest.mark.parametrize(
    "uri",
    [
        MLEM_TEST_REPO_URI + "/src/main/path",
        f"bitbucket://{MLEM_TEST_REPO_PROJECT}@main/path",
    ],
)
def test_uri_resolver(uri):
    fs, path = get_fs(uri)

    assert isinstance(fs, BitBucketFileSystem)
    assert path == "path"


@long
@pytest.mark.parametrize(
    "rev",
    ["main", "branch", "tag", "eb2ffd48b624589ce8e39b51a4b984f6887deeb5"],
)
def test_uri_resolver_rev(rev):
    location = UriResolver.resolve(MLEM_TEST_REPO_URI, None, rev=rev, fs=None)
    assert isinstance(location.fs, BitBucketFileSystem)
    assert location.fs.root == rev
    assert "README.md" in location.fs.ls("")


@long
def test_uri_resolver_wrong_rev():
    with pytest.raises(RevisionNotFound):
        UriResolver.resolve(
            MLEM_TEST_REPO_URI, None, rev="__not_exists__", fs=None
        )
