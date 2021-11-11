import os.path
from json import loads

import pytest
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.http import HTTPFileSystem
from fsspec.implementations.local import LocalFileSystem
from gcsfs import GCSFileSystem
from s3fs import S3FileSystem

from mlem import CONFIG
from mlem.core.meta_io import (
    get_fs,
    get_path_by_fs_path,
    get_path_by_repo_path_rev,
    read,
)
from tests.conftest import (
    MLEM_TEST_REPO,
    MLEM_TEST_REPO_NAME,
    MLEM_TEST_REPO_ORG,
    long,
    need_test_repo_auth,
    resource_path,
)


@pytest.mark.parametrize(
    "url_path_pairs",
    [
        (resource_path(__file__, "file.txt"), "a"),
        (
            f"github://{MLEM_TEST_REPO_ORG}:{MLEM_TEST_REPO_NAME}@main/README.md",
            "#",
        ),
        (
            f"https://github.com/{MLEM_TEST_REPO_ORG}/{MLEM_TEST_REPO_NAME}/README.md",
            "#",
        ),
    ],
)
def test_read(url_path_pairs):
    uri, start = url_path_pairs
    assert read(uri).startswith(start)


@long
@pytest.mark.parametrize(
    "uri, cls, result",
    [
        ("path", LocalFileSystem, lambda: os.path.abspath("path")),
        ("file://path", LocalFileSystem, lambda: os.path.abspath("path")),
        ("s3://path", S3FileSystem, "path"),
        ("gcs://path", GCSFileSystem, "path"),
        # ("az://path", AzureBlobFileSystem),  # TODO: need credentials
        # TODO: see below in test_get_path_by_fs_path
        # (
        #     f"git://{os.path.abspath(__file__)}",
        #     GitFileSystem,
        #     "git:/" + __file__,
        # ),
        ("https://path", HTTPFileSystem, "https://path"),
    ],
)
def test_get_fs(uri, cls, result):
    if callable(result):
        result = result()
    fs, path = get_fs(uri)
    assert isinstance(fs, cls)
    assert path == result


@pytest.mark.parametrize(
    "uri, rev", [("path", "main"), ("tree/main/path", "main")]
)
def test_get_fs_github(uri, rev):
    fs, path = get_fs(os.path.join(MLEM_TEST_REPO, uri))
    assert isinstance(fs, GithubFileSystem)
    assert fs.org == MLEM_TEST_REPO_ORG
    assert fs.repo == MLEM_TEST_REPO_NAME
    assert fs.root == rev
    assert path == "path"


@pytest.mark.parametrize(
    "uri",
    [
        ("path", lambda: "file://" + os.path.abspath("path")),
        ("file://path", lambda: "file://" + os.path.abspath("path")),
        "s3://path",
        "gcs://path",
        # "az://path",  # TODO: need credentials
        # TODO: path after git:// does not goes to fs __init__ method for some reason in
        #  fsspec.core:640 for some reason, so test fails because it uses
        #  os.getcwd and it is not git repo in testing env
        # (f"git://{os.path.abspath(__file__)}", "git:/" + __file__),
        "https://path",
    ],
)
def test_get_path_by_fs_path(uri):
    if isinstance(uri, tuple):
        uri, result = uri
    else:
        result = uri
    if callable(result):
        result = result()
    uri2 = get_path_by_fs_path(*get_fs(uri))
    assert uri2 == result


@need_test_repo_auth
def test_get_path_by_fs_path_github():
    fs = GithubFileSystem(
        org=MLEM_TEST_REPO_ORG,
        repo=MLEM_TEST_REPO_NAME,
        sha="main",
        username=CONFIG.GITHUB_USERNAME,
        token=CONFIG.GITHUB_TOKEN,
    )
    uri = get_path_by_fs_path(fs, "path")
    fs2, path = get_fs(uri)
    assert loads(fs2.to_json()) == loads(fs.to_json())
    assert path == "path"


@pytest.mark.parametrize(
    "repo, rev, result",
    [
        (MLEM_TEST_REPO, None, os.path.join(MLEM_TEST_REPO, "path/file")),
        (
            MLEM_TEST_REPO,
            "branch",
            os.path.join(MLEM_TEST_REPO, "tree", "branch", "path/file"),
        ),
        (
            "git:///other/path",
            "branch",
            ("git:///other/path/path/file", {"rev": "branch"}),
        ),
    ],
)
def test_get_path_by_repo_path_rev(repo, rev, result):
    if isinstance(result, str):
        result = result, {}
    assert get_path_by_repo_path_rev(repo, "path/file", rev) == result
