from mlem.utils.github import ls_branches, ls_github_branches
from tests.conftest import (
    MLEM_TEST_REPO,
    need_test_repo_auth,
    need_test_repo_ssh_auth,
)


@need_test_repo_ssh_auth
def test_ls_branches():
    assert "main" in ls_branches(MLEM_TEST_REPO)


@need_test_repo_auth
def test_ls_github_branches():
    assert "main" in ls_github_branches("iterative", "mlem-test")
