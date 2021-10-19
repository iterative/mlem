import os.path
import random
import string

import pytest

from mlem.core.artifacts import FSSpecArtifact, FSSpecStorage
from mlem.core.meta_io import get_fs
from tests.conftest import resource_path

S3_TEST_BUCKET = "mlem-tests"


@pytest.fixture
def s3_tmp_filepath():
    fs, path = get_fs(
        f"s3://{S3_TEST_BUCKET}/"
        + "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    )
    yield os.path.basename(path)
    fs.delete(path)


def test_fsspec_backend_s3(tmpdir, s3_tmp_filepath):
    storage = FSSpecStorage(uri=f"s3://{S3_TEST_BUCKET}/", storage_options={})
    target = s3_tmp_filepath
    resource = resource_path(__file__, "file.txt")
    artifact = storage.upload(resource, target)
    assert isinstance(artifact, FSSpecArtifact)
    local_target = str(tmpdir / "file.txt")
    artifact.download(local_target)
    with open(local_target, "r", encoding="utf8") as actual, open(
        resource, "r", encoding="utf8"
    ) as expected:
        assert actual.read() == expected.read()
