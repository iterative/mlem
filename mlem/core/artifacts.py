"""
Artifacts which come with models and datasets,
such as model binaries or .csv files
"""
import contextlib
import os.path
from abc import ABC, abstractmethod
from typing import IO, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.local import LocalFileSystem

from mlem.core.base import MlemObject


class Artifact(MlemObject, ABC):
    __type_root__ = True
    __default_type__: ClassVar = "local"
    abs_name: ClassVar = "artifact"

    @abstractmethod
    def download(self, target_path: str) -> "LocalArtifact":
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        raise NotImplementedError

    def relative(
        self,
        fs: AbstractFileSystem,  # pylint: disable=unused-argument
        path: str,  # pylint: disable=unused-argument
    ) -> "Artifact":
        # TODO: maybe change fs and path to meta_storage in the future
        return self


class FSSpecArtifact(Artifact):
    type: ClassVar = "fsspec"
    uri: str

    def download(self, target_path: str) -> "LocalArtifact":
        from mlem.core.meta_io import get_fs

        fs, path = get_fs(self.uri)

        if os.path.isdir(target_path):
            target_path = os.path.join(target_path, os.path.basename(path))
        fs.download(path, target_path)
        return LocalArtifact(uri=target_path)

    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        from mlem.core.meta_io import get_fs

        fs, path = get_fs(self.uri)
        with fs.open(path) as f:
            yield f


class Storage(MlemObject, ABC):
    __type_root__ = True
    abs_name: ClassVar = "storage"

    def relative(
        self,
        fs: AbstractFileSystem,  # pylint: disable=unused-argument
        path: str,  # pylint: disable=unused-argument
    ) -> "Storage":
        return self

    @abstractmethod
    def upload(self, local_path: str, target_path: str) -> Artifact:
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, Artifact]]:
        raise NotImplementedError


class FSSpecStorage(Storage):
    type: ClassVar = "fsspec"

    __transient_fields__: ClassVar = {"fs", "base_path"}

    fs: ClassVar[Optional[AbstractFileSystem]] = None
    base_path: ClassVar[str] = ""
    uri: str
    storage_options: Optional[Dict[str, str]] = {}

    def upload(self, local_path: str, target_path: str) -> FSSpecArtifact:
        self.get_fs().upload(
            local_path, os.path.join(self.base_path, target_path)
        )
        return FSSpecArtifact(uri=self.create_uri(target_path))

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, FSSpecArtifact]]:
        fs = self.get_fs()
        fullpath = os.path.join(self.base_path, path)
        fs.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with fs.open(fullpath, "wb") as f:
            yield f, FSSpecArtifact(uri=(self.create_uri(path)))

    def create_uri(self, path):
        uri = os.path.join(self.uri, path)
        if os.path.isabs(path):
            protocol = urlparse(self.uri).scheme or "file"
            uri = f"{protocol}://{uri}"
        return uri

    def get_fs(self) -> AbstractFileSystem:
        if self.fs is None:
            self.fs, _, (self.base_path,) = fsspec.get_fs_token_paths(
                self.uri, storage_options=self.storage_options
            )
        return self.fs


class LocalStorage(FSSpecStorage):
    type: ClassVar = "local"
    fs = LocalFileSystem()

    def relative(self, fs: AbstractFileSystem, path: str) -> "Storage":
        if isinstance(fs, LocalFileSystem):
            return LocalStorage(uri=self.create_uri(path))
        protocol = fs.protocol
        if isinstance(protocol, list):
            protocol = protocol[0]
        storage = FSSpecStorage(uri=f"{protocol}://{path}")
        storage.fs = fs
        storage.base_path = self.create_uri(path)
        return storage

    def upload(self, local_path: str, target_path: str) -> "LocalArtifact":
        return LocalArtifact(uri=super().upload(local_path, target_path).uri)

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, "LocalArtifact"]]:
        with super().open(os.path.join(self.uri, path)) as (io, _):
            yield io, LocalArtifact(uri=path)


class LocalArtifact(FSSpecArtifact):
    type: ClassVar = "local"

    def relative(self, fs: AbstractFileSystem, path: str) -> "FSSpecArtifact":

        if isinstance(fs, LocalFileSystem):
            return LocalArtifact(uri=os.path.join(path, self.uri))

        return FSSpecArtifact(
            uri=get_path_by_fs_path(fs, os.path.join(path, self.uri))
        )


def get_path_by_fs_path(fs: AbstractFileSystem, path: str):
    """Restore full uri from fs and path

    Not ideal, but alternative to this is to save uri on MlemMeta level and pass it everywhere
    Another alternative is to support this on fsspec level, but we need to contribute it ourselves"""
    if isinstance(fs, GithubFileSystem):
        # here "rev" should be already url encoded
        return f"{fs.protocol}://{fs.org}:{fs.repo}@{fs.root}/{path}"
    protocol = fs.protocol
    if isinstance(protocol, list):
        protocol = protocol[0]
    return f"{protocol}://{path}"


LOCAL_STORAGE = LocalStorage(uri="")

Artifacts = List[Artifact]
