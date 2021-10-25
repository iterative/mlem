import subprocess
from pathlib import Path

import git
import pytest
from git import Repo

MLEM_TEST_REPO = "https://github.com/iterative/mlem-test/"


def _check_github_test_repo_auth():
    try:
        git.cmd.Git().ls_remote(MLEM_TEST_REPO)
        return True
    except subprocess.CalledProcessError:
        return False


need_test_repo_auth = pytest.mark.skipif(
    not _check_github_test_repo_auth(), reason="No credentials for remote repo"
)


@pytest.fixture()
def current_test_branch():
    branch = Repo(str(Path(__file__).parent.parent.parent)).active_branch
    g = git.cmd.Git()
    remote_refs = {
        ref.split("\t")[1] for ref in g.ls_remote(MLEM_TEST_REPO).split("\n")
    }
    if branch.path in remote_refs:
        return branch.name
    return "main"
