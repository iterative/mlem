"""
Utils functions that parse and process supplied URI, serialize/derialize MLEM objects
"""
import contextlib
import posixpath
from abc import ABC, abstractmethod
from inspect import isabstract
from typing import List, Optional, Tuple, Type

from fsspec import AbstractFileSystem, get_fs_token_paths
from fsspec.implementations.github import GithubFileSystem
from pydantic import BaseModel

from mlem.core.errors import MlemObjectNotFound
from mlem.utils.github import get_github_envs, get_github_kwargs
from mlem.utils.root import MLEM_DIR, find_repo_root

MLEM_EXT = ".mlem"


class Location(BaseModel):
    path: str
    repo: Optional[str]
    rev: Optional[str]
    uri: str
    fs: AbstractFileSystem

    class Config:
        arbitrary_types_allowed = True

    @property
    def fullpath(self):
        return posixpath.join(self.repo or "", self.path)

    @property
    def path_in_repo(self):
        return posixpath.relpath(self.fullpath, self.repo)

    @property
    def repo_uri(self):
        if self.repo is None:
            return None
        # not sure if this is ok
        # maybe we need to merge Location with UriResolver and implement this separately for each case
        return self.uri[: -len(self.path)]

    @contextlib.contextmanager
    def open(self, mode="r", **kwargs):
        with self.fs.open(self.fullpath, mode, **kwargs) as f:
            yield f

    @classmethod
    def abs(cls, path: str, fs: AbstractFileSystem):
        return Location(path=path, repo=None, fs=fs, uri=path)

    def update_path(self, path):
        if not self.uri.endswith(self.path):
            raise ValueError("cannot automatically update uri")
        self.uri = self.uri[: -len(self.path)] + path
        self.path = path


class UriResolver(ABC):
    impls: List[Type["UriResolver"]] = []
    versioning_support: bool = False

    def __init_subclass__(cls, *args, **kwargs):
        if not isabstract(cls) and cls not in cls.impls:
            cls.impls.append(cls)
        super(UriResolver, cls).__init_subclass__(*args, **kwargs)

    @classmethod
    def resolve(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
        find_repo: bool = False,
    ) -> Location:
        return cls.find_resolver(path, repo, rev, fs).process(
            path, repo, rev, fs, find_repo
        )

    @classmethod
    def find_resolver(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> Type["UriResolver"]:
        for i in cls.impls:
            if i.check(path, repo, rev, fs):
                return i
        raise ValueError("No valid UriResolver implementation found")

    @classmethod
    @abstractmethod
    def check(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_fs(
        cls, uri: str, rev: Optional[str]
    ) -> Tuple[AbstractFileSystem, str]:
        raise NotImplementedError

    @classmethod
    def process(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
        find_repo: bool,
    ) -> Location:
        path, repo, rev, fs = cls.pre_process(path, repo, rev, fs)
        if rev is not None and not cls.versioning_support:
            raise ValueError(
                f"Rev {rev} was provided, but {cls} does not support versioning"
            )
        if fs is None:
            if repo is not None:
                fs, repo = cls.get_fs(repo, rev)
            else:
                fs, path = cls.get_fs(path, rev)
        if repo is None and find_repo:
            path, repo = cls.get_repo(path, fs)
        return Location(
            path=path,
            repo=repo,
            rev=rev,
            uri=cls.get_uri(path, repo, rev, fs),
            fs=fs,
        )

    @classmethod
    @abstractmethod
    def get_uri(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: AbstractFileSystem,
    ):
        raise NotImplementedError

    @classmethod
    def pre_process(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> Tuple[
        str, Optional[str], Optional[str], Optional[AbstractFileSystem]
    ]:
        return path, repo, rev, fs

    @classmethod
    def get_repo(
        cls, path: str, fs: AbstractFileSystem
    ) -> Tuple[str, Optional[str]]:
        repo = find_repo_root(path, fs, raise_on_missing=False)
        if repo is not None:
            path = posixpath.relpath(path, repo)
        return path, repo


class GithubResolver(UriResolver):
    PROTOCOL = "github://"
    GITHUB_COM = "https://github.com"

    # TODO: support on-prem github (other hosts)
    PREFIXES = [GITHUB_COM, PROTOCOL]
    versioning_support = True

    @classmethod
    def check(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> bool:
        fullpath = posixpath.join(repo or "", path)
        return isinstance(fs, GithubFileSystem) or any(
            fullpath.startswith(h) for h in cls.PREFIXES
        )

    @classmethod
    def get_fs(
        cls, uri: str, rev: Optional[str]
    ) -> Tuple[GithubFileSystem, str]:
        options = get_github_envs()
        if not uri.startswith(cls.PROTOCOL):
            options.update(get_github_kwargs(uri))
            uri = options.pop("path")
            options["sha"] = rev or options.get("sha", None)

        fs, _, (path,) = get_fs_token_paths(
            uri, protocol="github", storage_options=options
        )
        return fs, path

    @classmethod
    def get_uri(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: GithubFileSystem,
    ):
        fullpath = posixpath.join(repo or "", path)
        return (
            f"https://github.com/{fs.org}/{fs.repo}/tree/{fs.root}/{fullpath}"
        )

    @classmethod
    def pre_process(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ):
        if fs is not None and not isinstance(fs, GithubFileSystem):
            raise TypeError(
                f"{path, repo, rev, fs} cannot be resolved by {cls}: fs should be GithubFileSystem, not {fs.__class__}"
            )
        if (
            isinstance(fs, GithubFileSystem)
            and rev is not None
            and fs.root != rev
        ):
            fs.root = rev
            fs.invalidate_cache()

        return path, repo, rev, fs


class FSSpecResolver(UriResolver):
    @classmethod
    def check(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> bool:
        return True

    @classmethod
    def get_fs(
        cls, uri: str, rev: Optional[str]
    ) -> Tuple[AbstractFileSystem, str]:
        fs, _, (path,) = get_fs_token_paths(uri)
        return fs, path

    @classmethod
    def get_uri(
        cls,
        path: str,
        repo: Optional[str],
        rev: Optional[str],
        fs: AbstractFileSystem,
    ):
        fullpath = posixpath.join(repo or "", path)
        protocol = fs.protocol
        if isinstance(protocol, (tuple, list)):
            if any(fullpath.startswith(p) for p in protocol):
                return fullpath
            protocol = protocol[0]
        if fullpath.startswith(protocol):
            return fullpath
        return f"{protocol}://{fullpath}"


def get_fs(uri: str):
    location = UriResolver.resolve(path=uri, repo=None, rev=None, fs=None)
    return location.fs, location.fullpath


def get_path_by_fs_path(fs: AbstractFileSystem, path: str):
    """Restore full uri from fs and path

    Not ideal, but alternative to this is to save uri on MlemMeta level and pass it everywhere
    Another alternative is to support this on fsspec level, but we need to contribute it ourselves"""
    return UriResolver.find_resolver(path, None, None, fs=fs).get_uri(
        path, None, None, fs=fs
    )


def read(uri: str, mode: str = "r"):
    """Read file content by given path"""
    fs, path = get_fs(uri)
    with fs.open(path, mode=mode) as f:
        return f.read()


def get_meta_path(uri: str, fs: AbstractFileSystem) -> str:
    """Augments given path so it will point to a MLEM metafile
    if it points to a folder with dumped object
    """
    if uri.endswith(MLEM_EXT) and fs.isfile(uri):
        # .../<META_FILE_NAME>.<MLEM_EXT>
        return uri
    # if fs.isdir(uri) and fs.isfile(posixpath.join(uri, META_FILE_NAME)):
    #     # .../path and .../path/<META_FILE_NAME> exists
    #     return posixpath.join(uri, META_FILE_NAME)
    if fs.isfile(uri + MLEM_EXT):
        # .../name without <MLEM_EXT>
        return uri + MLEM_EXT
    if MLEM_DIR in uri and fs.isfile(uri):
        # .../<MLEM_DIR>/.../file
        return uri
    if fs.exists(uri):
        raise MlemObjectNotFound(
            f"{uri} is not a valid MLEM metafile or a folder with a MLEM model or dataset"
        )
    raise FileNotFoundError(uri)
