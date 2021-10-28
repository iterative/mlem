"""
Utils functions that parse and process supplied URI, serialize/derialize MLEM objects
"""
import os
from typing import Tuple, Type, TypeVar, Union

from fsspec import AbstractFileSystem, get_fs_token_paths
from fsspec.implementations.local import LocalFileSystem
from pydantic import parse_obj_as

from mlem.core.base import MlemObject
from mlem.utils.github import get_github_envs, get_github_kwargs
from mlem.utils.root import MLEM_DIR

MLEM_EXT = ".mlem.yaml"


META_FILE_NAME = "mlem.yaml"
ART_DIR = "artifacts"


def resolve_fs(
    fs: Union[str, AbstractFileSystem] = None, protocol: str = None
):
    """Try to resolve fs from given fs or URI"""
    # TODO: do we really need this function?
    # address in https://github.com/iterative/mlem/issues/4
    if fs is None:
        return LocalFileSystem()
    if isinstance(fs, AbstractFileSystem):
        return fs
    fs, _ = get_fs(uri=fs, protocol=protocol)
    return fs


def get_fs(
    uri: str, protocol: str = None, **kwargs
) -> Tuple[AbstractFileSystem, str]:
    """Parse given (uri, protocol) with fsspec and return (fs, path)"""
    storage_options = {}
    if protocol == "github" or uri.startswith("github://"):
        storage_options.update(get_github_envs())
    if protocol is None and uri.startswith("https://github.com"):
        protocol = "github"
        storage_options.update(get_github_envs())
        github_kwargs = get_github_kwargs(uri)
        uri = github_kwargs.pop("path")
        storage_options.update(github_kwargs)
    storage_options.update(kwargs)
    fs, _, (path,) = get_fs_token_paths(
        uri, protocol=protocol, storage_options=storage_options
    )
    return fs, path


def read(uri: str, mode: str = "r"):
    """Read file content by given path"""
    fs, path = get_fs(uri)
    with fs.open(path, mode=mode) as f:
        return f.read()


def serialize(
    obj, as_class: Type = None
):  # pylint: disable=unused-argument # todo remove later
    if not isinstance(obj, MlemObject):
        raise ValueError(f"{type(obj)} is not a subclass of MlemObject")
    return obj.dict(exclude_unset=True, exclude_defaults=True)


T = TypeVar("T")


def deserialize(obj, as_class: Type[T]) -> T:
    return parse_obj_as(as_class, obj)


def is_mlem_dir(uri: str, fs: AbstractFileSystem):
    """Check if given dir contains save MLEM model or dataset"""
    return fs.isdir(uri) and fs.exists(os.path.join(uri, META_FILE_NAME))


def get_meta_path(uri: str, fs: AbstractFileSystem) -> str:
    """Augments given path so it will point to a MLEM metafile
    if it points to a folder with dumped object
    """
    if os.path.basename(uri) == META_FILE_NAME and fs.isfile(uri):
        return uri
    if is_mlem_dir(uri, fs):
        return os.path.join(uri, META_FILE_NAME)
    if MLEM_DIR in uri and fs.isfile(uri):
        return uri
    if fs.exists(uri):
        raise Exception(
            f"{uri} is not a valid MLEM metafile or a folder with a MLEM model or dataset"
        )
    raise FileNotFoundError(uri)


# def blobs_from_path(path: str, fs: AbstractFileSystem = None):
#     if fs is None:
#         fs, path = get_fs(path)
#     file_list = fs.glob(f'{path}/*', recursive=True)
#     if fs.protocol == 'file':
#         return Blobs({os.path.relpath(name, path): RepoFileBlob(name) for name in file_list})
#     return Blobs({os.path.relpath(name, path): FSBlob(name, fs) for name in file_list})


# class RepoFileBlob(LocalFileBlob):
#     def __init__(self, path: str):
#         super().__init__(os.path.relpath(path, repo_root()))


# @dataclass
# class FSBlob(Blob, Unserializable):
#     path: str
#     fs: AbstractFileSystem
#
#     def materialize(self, path):
#         self.fs.get_file(self.path, path)
#
#     @contextlib.contextmanager
#     def bytestream(self) -> StreamContextManager:
#         with self.fs.open(self.path) as f:
#             yield f
