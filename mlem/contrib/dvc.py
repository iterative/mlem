"""DVC Support
Extension type: storage

Support for storing artifacts with DVC
"""
import contextlib
import os.path
import posixpath
import shutil
from typing import IO, ClassVar, Iterator, Tuple
from urllib.parse import unquote_plus

from fsspec import AbstractFileSystem
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.local import LocalFileSystem

from mlem.core.artifacts import (
    LocalArtifact,
    LocalStorage,
    Storage,
    get_local_file_info,
)
from mlem.core.meta_io import get_fs
from mlem.core.registry import ArtifactInRegistry

BATCH_SIZE = 10**5


def find_dvc_repo_root(path: str):
    from dvc.exceptions import NotDvcRepoError

    _path = path[:]
    while True:
        if os.path.isdir(os.path.join(_path, ".dvc")):
            return _path
        if _path == "/" or not _path:
            break
        _path = os.path.dirname(_path)
    raise NotDvcRepoError(f"Path {path} is not in dvc repo")


class DVCStorage(LocalStorage):
    """User-managed dvc storage, which means user should
    track corresponding files with dvc manually."""

    #  TODO: https://github.com//issues/47

    type: ClassVar = "dvc"
    uri: str = ""
    """Base storage path"""

    def upload(self, local_path: str, target_path: str) -> "DVCArtifact":
        return DVCArtifact(
            uri=super().upload(local_path, target_path).uri,
            **get_local_file_info(local_path),
        )

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, "DVCArtifact"]]:
        with super().open(path) as (io, art):
            dvc_art = DVCArtifact(uri=path, size=-1, hash="")
            yield io, dvc_art
        dvc_art.size = art.size
        dvc_art.hash = art.hash

    def relative(self, fs: AbstractFileSystem, path: str) -> Storage:
        storage = super().relative(fs, path)
        if isinstance(storage, LocalStorage):
            return DVCStorage(uri=storage.uri)  # pylint: disable=no-member
        return storage


class DVCArtifact(LocalArtifact):
    """Local artifact that can be also read from DVC cache"""

    type: ClassVar = "dvc"
    uri: str
    """Local path to file"""

    def _download(self, target_path: str) -> LocalArtifact:
        if os.path.isdir(target_path):
            target_path = posixpath.join(
                target_path, os.path.basename(self.uri)
            )
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with self.open() as fin, open(target_path, "wb") as fout:
            shutil.copyfileobj(fin, fout, BATCH_SIZE)
        return LocalArtifact(uri=target_path, size=self.size, hash=self.hash)

    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        from dvc.api import open

        fs, path = get_fs(self.uri)
        # TODO: support other sources of dvc-tracked repos
        #  At least local and git
        if isinstance(fs, GithubFileSystem):
            with open(
                path,
                f"https://github.com/{fs.org}/{fs.repo}",
                unquote_plus(fs.root),
                mode="rb",
            ) as f:
                yield f
                return
        elif isinstance(fs, LocalFileSystem):
            if not os.path.exists(path):
                root = find_dvc_repo_root(path)
                # alternative caching impl
                # Repo(root).pull(os.path.relpath(path, root))
                with open(
                    os.path.relpath(path, root), repo=root, mode="rb"
                ) as f:
                    yield f
                    return
        with fs.open(path) as f:
            yield f

    def relative(self, fs: AbstractFileSystem, path: str) -> "DVCArtifact":
        relative = super().relative(fs, path)
        return DVCArtifact(uri=relative.uri, size=self.size, hash=self.hash)


def find_artifact_name_by_path(path):
    if not os.path.abspath(path):
        raise ValueError(f"Path {path} is not absolute")

    from dvc.repo import Repo

    root = find_dvc_repo_root(path)
    relpath = os.path.relpath(path, root)
    if relpath.endswith(".mlem"):
        relpath = relpath[:-5]
    repo = Repo(root)
    for _dvcyaml, artifacts in repo.artifacts.read().items():
        for name, value in artifacts.items():
            if value.path == relpath:
                return root, name
    return root, None


def find_version(root, name):
    from gto.api import _show_versions

    version = _show_versions(root, name, ref="HEAD")
    if version:
        return version[0]["version"]


class DVCArtifactInRegistry(ArtifactInRegistry):
    """Artifact registered within an Artifact Registry."""

    type: ClassVar = "dvc"
    uri: str
    """Local path to file"""

    @property
    def version(self):
        root, name = find_artifact_name_by_path(self.uri)
        if name:
            return find_version(root, name)
