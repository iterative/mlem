import os.path

import pytest
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

from mlem.core.artifacts import (
    FSSpecArtifact,
    FSSpecStorage,
    LocalArtifact,
    LocalStorage,
)
from tests.conftest import long, resource_path


@long
def test_fsspec_backend_s3_upload(tmpdir, s3_tmp_path, s3_storage):
    target = os.path.basename(s3_tmp_path("upload"))
    resource = resource_path(__file__, "file.txt")
    artifact = s3_storage.upload(resource, target)
    assert isinstance(artifact, FSSpecArtifact)
    assert artifact.hash != ""
    assert artifact.size > 0
    local_target = str(tmpdir / "file.txt")
    artifact.materialize(local_target)
    with open(local_target, "r", encoding="utf8") as actual, open(
        resource, "r", encoding="utf8"
    ) as expected:
        assert actual.read() == expected.read()


@long
def test_fsspec_backend_s3_open(s3_tmp_path, s3_storage):
    target = os.path.basename(s3_tmp_path("open"))
    with s3_storage.open(target) as (f, artifact):
        f.write(b"a")
    assert isinstance(artifact, FSSpecArtifact)
    assert artifact.hash != ""
    assert artifact.size > 0
    with artifact.open() as f:
        assert f.read() == b"a"


@pytest.mark.parametrize("fs", [LocalFileSystem(), S3FileSystem()])
def test_relative_storage_remote(fs):
    """This checks that if artifact path is absolute,
    it will stay that way if meta is stored locally or remotely.
    """
    s3storage = FSSpecStorage(uri="s3://some_bucket")
    rel1 = s3storage.relative(fs, "some_path")
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


def test_local_storage_relative(tmpdir):
    storage = LocalStorage(uri=str(tmpdir))
    rstorage = storage.relative(LocalFileSystem(), "subdir")
    with rstorage.open("file2") as (f, open_art):
        f.write(b"1")
    assert isinstance(open_art, LocalArtifact)
    assert open_art.hash != ""
    assert open_art.size > 0
    assert open_art.uri == "file2"
    assert os.path.isfile(os.path.join(tmpdir, "subdir", open_art.uri))

    upload_art = rstorage.upload(__file__, "file")
    assert isinstance(upload_art, LocalArtifact)
    assert upload_art.uri == "file"
    assert upload_art.hash != ""
    assert upload_art.size > 0
    assert os.path.isfile(os.path.join(tmpdir, "subdir", upload_art.uri))
