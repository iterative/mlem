import os

import pytest

from mlem.core.artifacts import FSSpecStorage
from mlem.core.meta_io import get_fs

S3_TEST_BUCKET = "mlem-tests"


@pytest.fixture
def s3_tmp_path():
    paths = set()
    base_path = f"s3://{S3_TEST_BUCKET}"
    fs, _ = get_fs(base_path)

    def gen(path):
        path = os.path.join(base_path, path)
        if path in paths:
            raise ValueError(f"Already generated {path}")
        if fs.exists(path):
            fs.delete(path, recursive=True)
        paths.add(path)
        return path

    yield gen
    for path in paths:
        fs.delete(path, recursive=True)


@pytest.fixture()
def s3_storage():
    return FSSpecStorage(uri=f"s3://{S3_TEST_BUCKET}/", storage_options={})


@pytest.fixture()
def s3_storage_fs(s3_storage):
    return s3_storage.get_fs()
