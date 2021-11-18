import os
from typing import Optional, overload

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from typing_extensions import Literal

from mlem.constants import MLEM_DIR
from mlem.core.errors import MlemRootNotFound


@overload
def find_mlem_root(
    path: str = ".",
    fs: AbstractFileSystem = None,
    raise_on_missing: Literal[True] = ...,
    recursive: bool = True,
) -> str:
    ...


@overload
def find_mlem_root(
    path: str = ".",
    fs: AbstractFileSystem = None,
    raise_on_missing: Literal[False] = ...,
    recursive: bool = True,
) -> Optional[str]:
    ...


def find_mlem_root(
    path: str = ".",
    fs: AbstractFileSystem = None,
    raise_on_missing: bool = True,
    recursive: bool = True,
) -> Optional[str]:
    """Search for mlem root folder, starting from the given path
    and up the directory tree.
    Raises an Exception if folder is not found.
    """
    if fs is None:
        fs = LocalFileSystem()
    if isinstance(fs, LocalFileSystem):
        path = os.path.abspath(path)
    _path = path[:]
    if not recursive:
        if fs.exists(os.path.join(_path, MLEM_DIR)):
            return _path
    else:
        if fs.isfile(_path) or not fs.exists(_path):
            _path = os.path.dirname(_path)
        while True:
            if fs.exists(os.path.join(_path, MLEM_DIR)):
                return _path
            if _path == os.path.dirname(_path):
                break

            _path = os.path.dirname(_path)
    if raise_on_missing:
        raise MlemRootNotFound(path, fs)
    return None
