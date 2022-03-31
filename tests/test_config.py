import posixpath

from mlem.config import CONFIG_FILE_NAME, MlemConfig, repo_config
from mlem.constants import MLEM_DIR
from mlem.core.artifacts import FSSpecStorage, LocalStorage
from mlem.core.meta_io import get_fs
from tests.conftest import long


def test_loading_storage(set_mlem_repo_root):
    set_mlem_repo_root("storage")
    config = MlemConfig()
    assert config.ADDITIONAL_EXTENSIONS == ["ext1"]
    assert config.default_storage == FSSpecStorage(uri="s3://somebucket")


def test_loading_empty(set_mlem_repo_root):
    set_mlem_repo_root("empty")
    config = MlemConfig()
    assert isinstance(config.default_storage, LocalStorage)


@long
def test_loading_remote(s3_tmp_path, s3_storage_fs):
    repo = s3_tmp_path("remote_conf")
    fs, path = get_fs(repo)
    path = posixpath.join(path, MLEM_DIR, CONFIG_FILE_NAME)
    with fs.open(path, "w") as f:
        f.write("ADDITIONAL_EXTENSIONS_RAW: ext1\n")
    assert repo_config(path, fs=fs).ADDITIONAL_EXTENSIONS == ["ext1"]
