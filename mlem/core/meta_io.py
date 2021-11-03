"""
Utils functions that parse and process supplied URI, serialize/derialize MLEM objects
"""
import os
from typing import Any, Dict, Optional, Tuple, Type, TypeVar, Union

from fsspec import AbstractFileSystem, get_fs_token_paths
from fsspec.implementations.github import GithubFileSystem
from fsspec.implementations.local import LocalFileSystem
from pydantic import parse_obj_as

from mlem.core.base import MlemObject
from mlem.utils.github import get_github_envs, get_github_kwargs
from mlem.utils.root import MLEM_DIR

MLEM_EXT = ".mlem.yaml"


META_FILE_NAME = "mlem.yaml"
ART_DIR = "artifacts"


def resolve_fs(
    fs: Union[str, AbstractFileSystem] = None,
    path: str = None,
    protocol: str = None,
) -> Tuple[AbstractFileSystem, Optional[str]]:
    """Try to resolve fs from given fs or URI"""
    # TODO: do we really need this function?
    # address in https://github.com/iterative/mlem/issues/4
    if fs is None:
        if path is None:
            return LocalFileSystem(), ""
        return get_fs(path)
    if isinstance(fs, AbstractFileSystem):
        return fs, path
    return get_fs(uri=fs, protocol=protocol)


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


def get_path_by_fs_path(fs: AbstractFileSystem, path: str):
    """Restore full uri from fs and path

    Not ideal, but alternative to this is to save uri on MlemMeta level and pass it everywhere
    Another alternative is to support this on fsspec level, but we need to contribute it ourselves"""
    if isinstance(fs, GithubFileSystem):
        # here "rev" should be already url encoded
        return f"{fs.protocol}://{fs.org}:{fs.repo}@{fs.root}/{path}"
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        if any(path.startswith(p) for p in protocol):
            return path
        protocol = protocol[0]
    if path.startswith(f"{protocol}://"):
        return path
    return f"{protocol}://{path}"


def get_path_by_repo_path_rev(
    repo: str, path: str, rev: str = None
) -> Tuple[str, Dict[str, Any]]:
    """Construct uri from repo url, relative path in repo and optional revision.
    Also returns additional kwargs for fs"""
    if repo.startswith("https://github.com"):
        if rev is None:
            # https://github.com/org/repo/path
            return os.path.join(repo, path), {}
        # https://github.com/org/repo/tree/branch/path
        return os.path.join(repo, "tree", rev, path), {}
    # TODO: do something about git protocol
    return os.path.join(repo, path), {"rev": rev}


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
    return obj.dict(exclude_unset=True, exclude_defaults=True, by_alias=True)


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
