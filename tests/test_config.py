import pytest

from mlem.config import MlemConfig
from mlem.core.artifacts import FSSpecStorage, LocalStorage
from tests.conftest import resource_path


@pytest.fixture
def set_mlem_repo_root(mocker):
    def set(path):
        mocker.patch(
            "mlem.utils.root.find_repo_root",
            return_value=resource_path(__file__, path),
        )

    return set


def test_loading_storage(set_mlem_repo_root):
    set_mlem_repo_root("storage")
    config = MlemConfig()
    assert config.ADDITIONAL_EXTENSIONS == ["ext1"]
    assert config.default_storage == FSSpecStorage(uri="s3://somebucket")


def test_loading_empty(set_mlem_repo_root):
    set_mlem_repo_root("empty")
    config = MlemConfig()
    assert isinstance(config.default_storage, LocalStorage)
