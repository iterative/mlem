import posixpath

from mlem.config import MlemConfig, project_config
from mlem.constants import MLEM_CONFIG_FILE_NAME
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.artifacts import FSSpecStorage, LocalStorage
from mlem.core.meta_io import get_fs
from tests.conftest import long


def test_loading_storage(set_mlem_project_root):
    set_mlem_project_root("storage")
    config = MlemConfig()
    assert config.additional_extensions == ["ext1"]
    assert config.storage == FSSpecStorage(uri="s3://somebucket")


def test_loading_empty(set_mlem_project_root):
    set_mlem_project_root("empty")
    config = MlemConfig()
    assert isinstance(config.storage, LocalStorage)


@long
def test_loading_remote(s3_tmp_path, s3_storage_fs):
    project = s3_tmp_path("remote_conf")
    fs, path = get_fs(project)
    path = posixpath.join(path, MLEM_CONFIG_FILE_NAME)
    with fs.open(path, "w") as f:
        f.write("core:\n  ADDITIONAL_EXTENSIONS: ext1\n")
    assert project_config(path, fs=fs).additional_extensions == ["ext1"]


def test_default_server():
    assert project_config("").server == FastAPIServer()
