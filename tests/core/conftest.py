import subprocess

import pytest


def _check_github_example_auth():
    try:
        subprocess.check_call(
            "git ls-remote https://github.com/iterative/example-mlem/",
            shell=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


need_example_auth = pytest.mark.skipif(
    not _check_github_example_auth(), reason="No credentials for remote repo"
)
