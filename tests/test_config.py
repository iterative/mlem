from mlem.config import MlemConfig
from mlem.core.artifacts import FSSpecStorage, LocalStorage


def test_loading_storage(set_mlem_repo_root):
    set_mlem_repo_root("storage")
    config = MlemConfig()
    assert config.ADDITIONAL_EXTENSIONS == ["ext1"]
    assert config.default_storage == FSSpecStorage(uri="s3://somebucket")


def test_loading_empty(set_mlem_repo_root):
    set_mlem_repo_root("empty")
    config = MlemConfig()
    assert isinstance(config.default_storage, LocalStorage)
