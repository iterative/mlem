from pathlib import Path

import git
import pytest
from git import GitCommandError, Repo
from requests import HTTPError

from mlem import CONFIG
from mlem.core.meta_io import get_fs
from mlem.utils.github import ls_remotes

MLEM_TEST_REPO = "https://github.com/iterative/mlem-test/"


def _check_github_test_repo_ssh_auth():
    try:
        git.cmd.Git().ls_remote(MLEM_TEST_REPO)
        return True
    except GitCommandError:
        return False


def _check_github_test_repo_auth():
    if not CONFIG.GITHUB_USERNAME or not CONFIG.GITHUB_TOKEN:
        return False
    try:
        get_fs(MLEM_TEST_REPO)
        return True
    except HTTPError:
        return False


need_test_repo_auth = pytest.mark.skipif(
    not _check_github_test_repo_auth(),
    reason="No http credentials for remote repo",
)

need_test_repo_ssh_auth = pytest.mark.skipif(
    not _check_github_test_repo_ssh_auth(),
    reason="No ssh credentials for remote repo",
)


@pytest.fixture()
def current_test_branch():
    branch = Repo(str(Path(__file__).parent.parent.parent)).active_branch
    remote_refs = set(ls_remotes(MLEM_TEST_REPO).keys())
    if branch.path in remote_refs:
        return branch.name
    return "main"
