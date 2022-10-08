import subprocess

import pytest

from mlem.contrib.docker.context import use_mlem_source
from tests.conftest import long


@pytest.fixture()
def uses_docker_build():
    with use_mlem_source("whl"):
        yield


def has_conda():
    assert subprocess.run(["conda"], check=True).returncode == 0


def conda_test(f):
    mark = pytest.mark.kubernetes
    skip = pytest.mark.skipif(not has_conda(), reason="conda is unavailable")
    return long(mark(skip(f)))
