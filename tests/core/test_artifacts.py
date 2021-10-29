import os.path
import random
import string

import pytest
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

from mlem.core.artifacts import FSSpecArtifact, FSSpecStorage, LocalStorage
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


def test_fsspec_backend_s3_upload(tmpdir, s3_tmp_filepath):
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


def test_fsspec_backend_s3_open(s3_tmp_filepath):
    storage = FSSpecStorage(uri=f"s3://{S3_TEST_BUCKET}/", storage_options={})
    target = s3_tmp_filepath
    with storage.open(target) as (f, artifact):
        f.write(b"a")
    assert isinstance(artifact, FSSpecArtifact)

    with artifact.open() as f:
        assert f.read() == b"a"


def test_relative_storage_remote():
    s3storage = FSSpecStorage(uri="s3://some_bucket")
    rel1 = s3storage.relative(LocalFileSystem(), "some_path")
    assert rel1 == s3storage


def test_relative_storage_local():
    """This test case covers a scenario when meta was stored in remote storage
    and then was downloaded to local storage, but artifacts are still in the remote.
    Then the relative path to artifacts would be the path in the remote.
    """
    local_storage = LocalStorage(uri="")
    rel1 = local_storage.relative(S3FileSystem(), "some_path")
    assert rel1 != local_storage
    assert isinstance(rel1, FSSpecStorage)
    assert rel1.uri == "s3://some_path"
