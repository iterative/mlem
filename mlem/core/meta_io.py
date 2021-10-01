import os
import pathlib
from typing import Dict, Tuple, Type, TypeVar, Union
from urllib.parse import urlparse

from fsspec import AbstractFileSystem, get_fs_token_paths
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import parse_obj_as

from mlem.config import CONFIG
from mlem.core.base import MlemObject
from mlem.utils.root import MLEM_DIR

MLEM_EXT = ".mlem.yaml"


META_FILE_NAME = "mlem.yaml"
ART_DIR = "artifacts"


def get_git_kwargs(uri: str):
    """Parse URI to git repo to get dict with all URI parts"""
    # TODO: do we lose URL to the site, like https://github.com?
    # should be resolved as part of https://github.com/iterative/mlem/issues/4
    parsed = urlparse(uri)
    parts = pathlib.Path(parsed.path).parts
    org, repo, *path = parts[1:]
    if not path:
        return {
            "org": org,
            "repo": repo,
        }
    if path[0] == "tree":
        sha = path[1]
        path = path[2:]
    else:
        sha = CONFIG.DEFAULT_BRANCH
    return {
        "org": org,
        "repo": repo,
        "sha": sha,
        "path": os.path.join(*path),
    }


def resolve_fs(
    fs: Union[str, AbstractFileSystem] = None, protocol: str = None
):
    """Try to resolve fs from given fs or URI"""
    # TODO: do we really need this function?
    # address in https://github.com/iterative/mlem/issues/4
    if fs is None:
        return LocalFileSystem()
    elif isinstance(fs, AbstractFileSystem):
        return fs
    fs, _ = get_fs(uri=fs, protocol=protocol)
    return fs


def get_envs() -> Dict:
    """Get authentification envs"""
    kwargs = {}
    if CONFIG.GITHUB_TOKEN is not None:
        kwargs["username"] = CONFIG.GITHUB_USERNAME
        kwargs["token"] = CONFIG.GITHUB_TOKEN
    return kwargs


def get_fs(uri: str, protocol: str = None) -> Tuple[AbstractFileSystem, str]:
    """Parse given (uri, protocol) with fsspec and return (fs, path)"""
    # kwargs = {}
    # if uri.startswith("https://github.com"):
    #     protocol = "github"
    #     uri, git_kwargs = _get_git_kwargs(uri)
    #     kwargs.update(git_kwargs)
    kwargs = get_envs()
    fs, _, (path,) = get_fs_token_paths(
        uri, protocol=protocol, storage_options=kwargs
    )
    return fs, path


def read(uri: str, mode: str = "r"):
    """Read file content by given path"""
    fs, path = get_fs(uri)
    with fs.open(path, mode=mode) as f:
        return f.read()


def serialize(obj, as_class: Type = None):
    if not isinstance(obj, MlemObject):
        raise ValueError(f"{type(obj)} is not a subclass of MlemObject")
    return obj.dict(exclude_unset=True)


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
    elif is_mlem_dir(uri, fs):
        return os.path.join(uri, META_FILE_NAME)
    elif MLEM_DIR in uri and fs.isfile(uri):
        return uri
    elif fs.exists(uri):
        raise Exception(
            f"{uri} is not a valid MLEM metafile or a folder with a MLEM model or dataset"
        )
    else:
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


def get_with_dvc(fs: GithubFileSystem, source_path, target_path):
    from dvc.repo import Repo

    repo_url = f"https://github.com/{fs.org}/{fs.repo}"
    Repo.get(repo_url, path=source_path, out=target_path, rev=fs.root)
