"""
Artifacts which come with models and datasets,
such as model binaries or .csv files
"""
import contextlib
import os.path
from abc import ABC, abstractmethod
from typing import IO, ClassVar, Dict, Iterator, List, Optional, Tuple

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from mlem.core.base import MlemObject


class Artifact(MlemObject, ABC):
    __type_root__ = True
    __default_type__: ClassVar = "local"

    @abstractmethod
    def download(
        self, target_path: str, repo_fs: AbstractFileSystem, repo_path: str
    ) -> "LocalArtifact":
        # TODO: maybe change fs and path to meta_storage in the future
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        raise NotImplementedError


class FSSpecArtifact(Artifact):
    type: ClassVar = "fsspec"
    uri: str

    def download(
        self, target_path: str, repo_fs: AbstractFileSystem, repo_path: str
    ) -> "LocalArtifact":
        from mlem.core.meta_io import get_fs

        fs, path = get_fs(os.path.join(repo_path, self.uri))
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
        return FSSpecArtifact(uri=os.path.join(self.uri, target_path))

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, FSSpecArtifact]]:
        fs = self.get_fs()
        fullpath = os.path.join(self.base_path, path)
        fs.makedirs(os.path.dirname(fullpath), exist_ok=True)
        with fs.open(fullpath, "wb") as f:
            yield f, FSSpecArtifact(uri=os.path.join(self.uri, path))

    def get_fs(self) -> AbstractFileSystem:
        if self.fs is None:
            self.fs, _, (self.base_path,) = fsspec.get_fs_token_paths(
                self.uri, storage_options=self.storage_options
            )
        return self.fs


class LocalStorage(FSSpecStorage):
    type: ClassVar = "local"
    fs = LocalFileSystem()

    def upload(self, local_path: str, target_path: str) -> "LocalArtifact":
        return LocalArtifact(uri=super().upload(local_path, target_path).uri)

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, "LocalArtifact"]]:
        with super().open(path) as (io, artifact):
            yield io, LocalArtifact(uri=artifact.uri)


class LocalArtifact(FSSpecArtifact):
    type: ClassVar = "local"


class DVCStorage(Storage):
    type: ClassVar = "dvc"


LOCAL_STORAGE = LocalStorage(uri="")

Artifacts = List[Artifact]
