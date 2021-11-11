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
from fsspec.implementations.local import LocalFileSystem

from mlem.core.base import MlemObject
from mlem.core.meta_io import get_fs, get_path_by_fs_path


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
        fs: AbstractFileSystem,
        path: str,
    ) -> "Artifact":
        # TODO: maybe change fs and path to meta_storage in the future
        raise NotImplementedError


class FSSpecArtifact(Artifact):
    type: ClassVar = "fsspec"
    uri: str

    def download(self, target_path: str) -> "LocalArtifact":
        fs, path = get_fs(self.uri)

        if os.path.isdir(target_path):
            target_path = os.path.join(target_path, os.path.basename(path))
        fs.download(path, target_path)
        return LocalArtifact(uri=target_path)

    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:

        fs, path = get_fs(self.uri)
        with fs.open(path) as f:
            yield f

    def relative(
        self,
        fs: AbstractFileSystem,
        path: str,
    ) -> "Artifact":
        return self


class Storage(MlemObject, ABC):
    __type_root__ = True
    abs_name: ClassVar = "storage"

    def relative(
        self,
        fs: AbstractFileSystem,
        path: str,
    ) -> "Storage":
        raise NotImplementedError

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
        fs = self.get_fs()
        path = os.path.join(self.base_path, target_path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        fs.upload(local_path, path)
        return FSSpecArtifact(uri=self.create_uri(target_path))

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, FSSpecArtifact]]:
        fs = self.get_fs()
        fullpath = os.path.join(self.base_path, path)
        fs.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with fs.open(fullpath, "wb") as f:
            yield f, FSSpecArtifact(uri=(self.create_uri(path)))

    def relative(
        self,
        fs: AbstractFileSystem,
        path: str,
    ) -> "Storage":
        return self

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

    @classmethod
    def from_fs_path(cls, fs: AbstractFileSystem, path: str):
        storage = cls(uri=get_path_by_fs_path(fs, path))
        storage.fs = fs
        # TODO: maybe wont work for github (but it does not support writing anyway)
        # pylint: disable=protected-access
        storage.base_path = fs._strip_protocol(path)
        return storage


class LocalStorage(FSSpecStorage):
    type: ClassVar = "local"
    fs = LocalFileSystem()

    @property
    def base_path(self):
        return self.uri

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
        super().upload(local_path, target_path)
        return LocalArtifact(uri=target_path)

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, "LocalArtifact"]]:
        with super().open(path) as (io, _):
            yield io, LocalArtifact(uri=path)


class LocalArtifact(FSSpecArtifact):
    type: ClassVar = "local"

    def relative(self, fs: AbstractFileSystem, path: str) -> "FSSpecArtifact":

        if isinstance(fs, LocalFileSystem):
            return LocalArtifact(uri=os.path.join(path, self.uri))

        return FSSpecArtifact(
            uri=get_path_by_fs_path(fs, os.path.join(path, self.uri))
        )


LOCAL_STORAGE = LocalStorage(uri="")

Artifacts = List[Artifact]
