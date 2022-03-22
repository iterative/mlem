import pytest

from mlem.contrib.docker.context import use_mlem_source


@pytest.fixture()
def uses_docker_build():
    with use_mlem_source("local"):
        yield
