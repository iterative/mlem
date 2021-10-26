"""
Artifacts which come with models and datasets,
such as model binaries or .csv files
"""
import contextlib
import os.path
from abc import ABC, abstractmethod
from typing import IO, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from mlem.core.base import MlemObject


class Artifact(MlemObject, ABC):
    __type_root__ = True
    __default_type__: ClassVar = "local"

    @abstractmethod
    def download(
        self,
        target_path: str
    ) -> "LocalArtifact":
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        raise NotImplementedError

    def relative(self,  storage: "Storage") -> "Artifact":
        return self

class FSSpecArtifact(Artifact):
    type: ClassVar = "fsspec"
    uri: str

    def download(
        self,
        target_path: str
    ) -> "LocalArtifact":
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
    uri: str

    def relative(self, storage: "Storage") -> "Storage":
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

    @property
    def base_path(self) -> str:
        return self.uri

    def relative(self, storage: Storage) -> "Storage":
        if isinstance(storage, LocalStorage):
            return LocalStorage(uri=os.path.join(storage.uri, self.uri))
        if isinstance(storage, FSSpecStorage):
            return FSSpecStorage(uri=os.path.join(storage.uri, self.uri))
        raise NotImplementedError

    def upload(self, local_path: str, target_path: str) -> "LocalArtifact":
        return LocalArtifact(uri=super().upload(local_path, target_path).uri)

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, "LocalArtifact"]]:
        with super().open(path) as (io, artifact):
            yield io, LocalArtifact(uri=path)


class LocalArtifact(FSSpecArtifact):
    type: ClassVar = "local"

    def relative(self, storage: Storage) -> "Artifact":

        uri = os.path.join(storage.uri, self.uri)
        if isinstance(storage, LocalStorage):
            return LocalArtifact(uri=uri)

        return FSSpecArtifact(uri=uri)


class DVCStorage(Storage):
    type: ClassVar = "dvc"


LOCAL_STORAGE = LocalStorage(uri="")

Artifacts = List[Artifact]
