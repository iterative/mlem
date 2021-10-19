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

from mlem.core.base import MlemObject


class Artifact(MlemObject, ABC):
    __type_root__ = True

    @abstractmethod
    def download(self, target_path: str):
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        raise NotImplementedError


class FSSpecArtifact(Artifact):
    type: ClassVar = "fsspec"
    uri: str

    def download(self, target_path: str):
        from mlem.core.meta_io import get_fs

        fs, path = get_fs(self.uri)
        fs.download(path, target_path)

    @contextlib.contextmanager
    def open(self) -> Iterator[IO]:
        from mlem.core.meta_io import get_fs

        fs, path = get_fs(self.uri)
        with fs.open(path) as f:
            yield f


class StorageBackend(MlemObject, ABC):
    __type_root__ = True

    @abstractmethod
    def upload(self, local_path: str, target_path: str) -> Artifact:
        raise NotImplementedError

    @abstractmethod
    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, Artifact]]:
        raise NotImplementedError


class FSSpecStorage(StorageBackend):
    type: ClassVar = "fsspec"

    __transient_fields__: ClassVar = {"fs", "base_path"}
    fs: ClassVar[Optional[AbstractFileSystem]] = None
    base_path: ClassVar[str] = ""
    uri: str
    storage_options: Optional[Dict[str, str]] = {}

    def upload(self, local_path: str, target_path: str) -> Artifact:
        self.get_fs().upload(
            local_path, os.path.join(self.base_path, target_path)
        )
        return FSSpecArtifact(uri=os.path.join(self.uri, target_path))

    @contextlib.contextmanager
    def open(self, path) -> Iterator[Tuple[IO, Artifact]]:
        with self.get_fs().open(os.path.join(self.base_path, path), "wb") as f:
            yield f, FSSpecArtifact(uri=os.path.join(self.uri, path))

    def get_fs(self) -> AbstractFileSystem:
        if self.fs is None:
            self.fs, _, (self.base_path,) = fsspec.get_fs_token_paths(
                self.uri, storage_options=self.storage_options
            )
        return self.fs


class DVCStorage(StorageBackend):
    type: ClassVar = "dvc"


LOCAL_STORAGE = FSSpecStorage(uri="")

Artifacts = List[str]
