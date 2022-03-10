import os.path

from mlem.api import pack
from mlem.contrib.fastapi import FastAPIServer
from mlem.pack.docker import DockerDirPackager
from mlem.pack.docker.context import DockerModelDirectory


def test_pack_dir(tmpdir, model_meta_saved):
    packed = pack(
        DockerDirPackager(server=FastAPIServer()),
        model_meta_saved,
        str(tmpdir),
    )
    assert isinstance(packed, DockerModelDirectory)
    assert os.path.isfile(tmpdir / "run.sh")
    assert os.path.isfile(tmpdir / "Dockerfile")
    assert os.path.isfile(tmpdir / "requirements.txt")
    assert os.path.isfile(tmpdir / "model")
    assert os.path.isfile(tmpdir / "model.mlem")


def test_pack_image():
    """TODO: https://github.com/iterative/mlem/issues/155"""
