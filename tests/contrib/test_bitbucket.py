import os

import pytest
from pytest_lazyfixture import lazy_fixture

from mlem.contrib.bitbucketfs import BitBucketFileSystem, ls_bb_refs
from mlem.core.errors import RevisionNotFound
from mlem.core.meta_io import Location, get_fs
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemModel
from tests.conftest import get_current_test_branch, long

MLEM_TEST_REPO_PROJECT = "iterative-ai/mlem-test"

MLEM_TEST_REPO_URI = f"https://bitbucket.org/{MLEM_TEST_REPO_PROJECT}"


@pytest.fixture()
def fs_no_auth():
    username = os.environ.get("BITBUCKET_USERNAME", None)
    try:
        del os.environ["BITBUCKET_USERNAME"]
        yield BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)
    finally:
        if username:
            os.environ["BITBUCKET_USERNAME"] = username


@pytest.fixture()
def fs_auth():
    return BitBucketFileSystem(MLEM_TEST_REPO_PROJECT)


@pytest.fixture()
def current_test_branch_bb():
    return get_current_test_branch(set(ls_bb_refs(MLEM_TEST_REPO_PROJECT)))


@long
@pytest.mark.parametrize(
    "fs",
    [lazy_fixture("fs_auth"), lazy_fixture("fs_no_auth")],
)
def test_ls(fs):
    assert "README.md" in fs.ls("")


@long
@pytest.mark.parametrize(
    "fs",
    [lazy_fixture("fs_auth"), lazy_fixture("fs_no_auth")],
)
def test_open(fs):
    with fs.open("README.md", "r") as f:
        assert f.read().startswith("### Fixture for mlem tests")


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
    ["main", "branch", "tag", "3897d2ab"],
)
def test_uri_resolver_rev(rev):
    location = Location.resolve(MLEM_TEST_REPO_URI, None, rev=rev, fs=None)
    assert isinstance(location.fs, BitBucketFileSystem)
    assert location.fs.root == rev
    assert "README.md" in location.fs.ls("")


@long
def test_uri_resolver_wrong_rev():
    with pytest.raises(RevisionNotFound):
        Location.resolve(
            MLEM_TEST_REPO_URI, None, rev="__not_exists__", fs=None
        )


@long
def test_loading_object(current_test_branch_bb):
    meta = load_meta(
        "latest",
        project=MLEM_TEST_REPO_URI + "/src/main/simple",
        rev=current_test_branch_bb,
    )
    assert isinstance(meta, MlemModel)
