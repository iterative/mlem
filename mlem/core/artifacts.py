"""
Artifacts which come with models and data,
such as model binaries or .csv files
"""
import contextlib
import hashlib
import os
import posixpath
import tempfile
from abc import ABC, abstractmethod
from typing import IO, ClassVar, Dict, Iterator, Optional, Tuple, overload
from urllib.parse import urlparse

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing_extensions import Literal, TypedDict

from mlem.core.base import MlemABC
from mlem.core.meta_io import Location, get_fs, get_path_by_fs_path
from mlem.utils.path import make_posix

CHUNK_SIZE = 2**20  # 1 mb


class ArtifactInfo(TypedDict):
    size: int
    hash: str


class Artifact(MlemABC, ABC):
    """Artifact represent a file in some storage. It can be opened for reading,
    downloaded to local fs or removed."""

    class Config:
        type_root = True
        default_type = "local"

    abs_name: ClassVar = "artifact"
    uri: str
    """location"""
    size: int
    """size in bytes"""
    hash: str
    """md5 hash"""

    @overload
    def materialize(
        self, target_path: str, target_fs: Literal[None] = None
    ) -> "LocalArtifact":
        ...

    @overload
    def materialize(
        self, target_path: str, target_fs: AbstractFileSystem = ...
    ) -> "Artifact":
        ...

    def materialize(
        self, target_path: str, target_fs: Optional[AbstractFileSystem] = None
    ) -> "Artifact":
        if target_fs is None:
            target_fs, target_path = get_fs(target_path)

        if isinstance(target_fs, LocalFileSystem):
            return self._download(target_path)
        with tempfile.TemporaryDirectory() as buf:
            tmp = self._download(buf)
            target_fs.upload(tmp.uri, target_path)
        return FSSpecArtifact(
            uri=get_path_by_fs_path(target_fs, target_path),
            size=self.size,
            hash=self.hash,
        )

    @abstractmethod
    def _download(self, target_path: str) -> "LocalArtifact":
        raise NotImplementedError

    @abstractmethod
    def remove(self):
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

    @property
    def info(self):
        return {"hash": self.hash, "size": self.size}


class FSSpecArtifact(Artifact):
    """Represents a file stored in an fsspec filesystem"""

    type: ClassVar = "fsspec"
    uri: str
    """Path to file"""

    def _download(self, target_path: str) -> "LocalArtifact":
        fs, path = get_fs(self.uri)

        if os.path.isdir(target_path):
            target_path = posixpath.join(target_path, posixpath.basename(path))
        LocalFileSystem().makedirs(
            posixpath.dirname(target_path), exist_ok=True
        )
        fs.download(path, target_path)
        return LocalArtifact(uri=target_path, **self.info)

    def remove(self):
        fs, path = get_fs(self.uri)
        fs.delete(path)

    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        fs, path = get_fs(self.uri)
        with fs.open(posixpath.normpath(path)) as f:
            yield f

    def relative(
        self,
        fs: AbstractFileSystem,
        path: str,
    ) -> "Artifact":
        return self


class PlaceholderArtifact(Artifact):
    """On dumping this artifact will be replaced with actual artifact that
    is relative to project root (if there is a project)"""

    type: ClassVar = "_placeholder"
    location: Location
    """Location of artifact"""

    def relative(self, fs: AbstractFileSystem, path: str) -> "Artifact":
        raise NotImplementedError

    def _download(self, target_path: str) -> "LocalArtifact":
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError

    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        raise NotImplementedError

    def relative_to(self, location: Location) -> "Artifact":
        if location.project is None:
            return FSSpecArtifact(uri=self.location.uri, **self.info)
        if self.location.fs == location.fs:
            return LocalArtifact(
                uri=posixpath.relpath(
                    self.location.fullpath,
                    posixpath.dirname(location.fullpath),
                ),
                **self.info,
            )
        return FSSpecArtifact(uri=self.uri, **self.info)


class Storage(MlemABC, ABC):
    """Storage represents a place where `Artifact`s can be stored. Storage can be
    used to upload local file or open a path in this storage for writing"""

    class Config:
        type_root = True

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
    """Represents an fsspec filesystem"""

    class Config:
        exclude = {"fs", "base_path"}
        arbitrary_types_allowed = True

    type: ClassVar = "fsspec"
    fs: Optional[AbstractFileSystem] = None
    base_path: str = ""
    uri: str
    """Path to storage dir"""
    storage_options: Optional[Dict[str, str]] = {}
    """Additional options for FS"""

    def upload(self, local_path: str, target_path: str) -> FSSpecArtifact:
        fs = self.get_fs()
        path = posixpath.join(self.get_base_path(), make_posix(target_path))
        fs.makedirs(posixpath.dirname(path), exist_ok=True)
        fs.upload(local_path, path)
        return FSSpecArtifact(
            uri=self.create_uri(target_path), **get_local_file_info(local_path)
        )

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, FSSpecArtifact]]:
        fs = self.get_fs()
        fullpath = posixpath.join(self.get_base_path(), path)
        fs.makedirs(posixpath.dirname(fullpath), exist_ok=True)
        art = FSSpecArtifact(uri=(self.create_uri(path)), size=-1, hash="")
        with fs.open(fullpath, "wb") as f:
            yield f, art
        file_info = get_file_info(fullpath, fs)
        art.size = file_info["size"]
        art.hash = file_info["hash"]

    def relative(
        self,
        fs: AbstractFileSystem,
        path: str,
    ) -> "Storage":
        return self

    def create_uri(self, path):
        uri = posixpath.join(self.uri, path)
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

    def get_base_path(self):
        return self.base_path

    @classmethod
    def from_fs_path(cls, fs: AbstractFileSystem, path: str):
        storage = cls(uri=get_path_by_fs_path(fs, path))
        storage.fs = fs
        # TODO: maybe wont work for github (but it does not support writing anyway)
        # pylint: disable=protected-access
        storage.base_path = fs._strip_protocol(path)
        return storage


class LocalStorage(FSSpecStorage):
    """Special case for local filesystem"""

    type: ClassVar = "local"
    fs: AbstractFileSystem = LocalFileSystem()

    def get_base_path(self):
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
        return LocalArtifact(
            uri=target_path, **get_local_file_info(local_path)
        )

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, "LocalArtifact"]]:
        with super().open(path) as (io, art):
            local_art = LocalArtifact(uri=path, size=-1, hash="")
            yield io, local_art
        local_art.size = art.size
        local_art.hash = art.hash


class LocalArtifact(FSSpecArtifact):
    """Special case for local file"""

    type: ClassVar = "local"

    def relative(self, fs: AbstractFileSystem, path: str) -> "FSSpecArtifact":

        if isinstance(fs, LocalFileSystem):
            return LocalArtifact(
                uri=posixpath.join(path, self.uri),
                size=self.size,
                hash=self.hash,
            )

        return FSSpecArtifact(
            uri=get_path_by_fs_path(fs, posixpath.join(path, self.uri)),
            size=self.size,
            hash=self.hash,
        )


def md5_fileobj(fobj):
    hash_md5 = hashlib.md5()  # nosec: B324
    for chunk in iter(lambda: fobj.read(CHUNK_SIZE), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_info(path: str, fs: AbstractFileSystem) -> ArtifactInfo:
    info = fs.info(path)
    with fs.open(path) as fobj:
        hash_value = md5_fileobj(fobj)
    return {"size": info["size"], "hash": hash_value}


def get_local_file_info(path: str):
    return get_file_info(path, LocalFileSystem())


LOCAL_STORAGE = LocalStorage(uri="")

Artifacts = Dict[str, Artifact]
