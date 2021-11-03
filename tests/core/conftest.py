import pytest

from mlem.core.artifacts import FSSpecStorage
from tests.conftest import MLEM_S3_TEST_BUCKET


@pytest.fixture()
def s3_storage():
    return FSSpecStorage(
        uri=f"s3://{MLEM_S3_TEST_BUCKET}/", storage_options={}
    )


@pytest.fixture()
def s3_storage_fs(s3_storage):
    return s3_storage.get_fs()
