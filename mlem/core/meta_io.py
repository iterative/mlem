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
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel

from mlem.core.errors import (
    HookNotFound,
    InvalidArgumentError,
    LocationNotFound,
    MlemObjectNotFound,
    RevisionNotFound,
)
from mlem.utils.github import (
    get_github_envs,
    get_github_kwargs,
    github_check_rev,
)
from mlem.utils.root import MLEM_DIR, find_project_root

MLEM_EXT = ".mlem"


class Location(BaseModel):
    path: str
    project: Optional[str]
    rev: Optional[str]
    uri: str
    project_uri: Optional[str]
    fs: AbstractFileSystem

    class Config:
        arbitrary_types_allowed = True

    @property
    def fullpath(self):
        return posixpath.join(self.project or "", self.path)

    @property
    def path_in_project(self):
        return posixpath.relpath(self.fullpath, self.project)

    @contextlib.contextmanager
    def open(self, mode="r", **kwargs):
        with self.fs.open(self.fullpath, mode, **kwargs) as f:
            yield f

    @classmethod
    def abs(cls, path: str, fs: AbstractFileSystem):
        return Location(
            path=path, project=None, fs=fs, uri=path, project_uri=None
        )

    def update_path(self, path):
        if not self.uri.endswith(self.path):
            raise ValueError("cannot automatically update uri")
        self.uri = self.uri[: -len(self.path)] + path
        self.path = path

    def exists(self):
        return self.fs.exists(self.fullpath)

    def is_same_project(self, other: "Location"):
        return other.fs == self.fs and other.project == self.project

    @property
    def uri_repr(self):
        if (
            isinstance(self.fs, LocalFileSystem)
            and posixpath.abspath("") in self.fullpath
        ):
            return posixpath.relpath(self.fullpath, "")
        return self.uri


class UriResolver(ABC):
    """Base class for resolving location. Turns (path, project, rev, fs) tuple
    into a normalized `Location` instance"""

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
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
        find_project: bool = False,
    ) -> Location:
        return cls.find_resolver(path, project, rev, fs).process(
            path, project, rev, fs, find_project
        )

    @classmethod
    def find_resolver(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> Type["UriResolver"]:
        for i in cls.impls:
            if i.check(path, project, rev, fs):
                return i
        raise HookNotFound("No valid UriResolver implementation found")

    @classmethod
    @abstractmethod
    def check(
        cls,
        path: str,
        project: Optional[str],
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
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
        find_project: bool,
    ) -> Location:
        path, project, rev, fs = cls.pre_process(path, project, rev, fs)
        if rev is not None and not cls.versioning_support:
            raise InvalidArgumentError(
                f"Rev `{rev}` was provided, but {cls.__name__} does not support versioning"
            )
        if fs is None:
            if project is not None:
                fs, project = cls.get_fs(project, rev)
            else:
                fs, path = cls.get_fs(path, rev)
        if project is None and find_project:
            path, project = cls.get_project(path, fs)
        uri = cls.get_uri(path, project, rev, fs)
        return Location(
            path=path,
            project=project,
            rev=rev,
            uri=uri,
            fs=fs,
            project_uri=cls.get_project_uri(path, project, rev, fs, uri),
        )

    @classmethod
    @abstractmethod
    def get_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: AbstractFileSystem,
    ):
        raise NotImplementedError

    @classmethod
    def pre_process(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> Tuple[
        str, Optional[str], Optional[str], Optional[AbstractFileSystem]
    ]:
        return path, project, rev, fs

    @classmethod
    def get_project(
        cls, path: str, fs: AbstractFileSystem
    ) -> Tuple[str, Optional[str]]:
        project = find_project_root(path, fs, raise_on_missing=False)
        if project is not None:
            path = posixpath.relpath(path, project)
        return path, project

    @classmethod
    def get_project_uri(  # pylint: disable=unused-argument
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: AbstractFileSystem,
        uri: str,
    ):
        if project is None:
            return None
        return uri[: -len(path)]


class GithubResolver(UriResolver):
    """Resolve https://github.com URLs"""

    PROTOCOL = "github://"
    GITHUB_COM = "https://github.com"

    # TODO: support on-prem github (other hosts)
    PREFIXES = [GITHUB_COM, PROTOCOL]
    versioning_support = True

    @classmethod
    def check(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> bool:
        fullpath = posixpath.join(project or "", path)
        return isinstance(fs, GithubFileSystem) or any(
            fullpath.startswith(h) for h in cls.PREFIXES
        )

    @classmethod
    def get_fs(
        cls, uri: str, rev: Optional[str]
    ) -> Tuple[GithubFileSystem, str]:
        options = get_github_envs()
        if not uri.startswith(cls.PROTOCOL):
            try:
                github_kwargs = get_github_kwargs(uri)
            except ValueError as e:
                raise LocationNotFound(*e.args) from e
            options.update(github_kwargs)
            path = options.pop("path")
            options["sha"] = rev or options.get("sha", None)
        else:
            path = uri
        try:
            fs, _, (path,) = get_fs_token_paths(
                path, protocol="github", storage_options=options
            )
        except FileNotFoundError as e:  # TODO catch HTTPError for wrong orgrepo
            if options["sha"] is not None and not github_check_rev(
                options["org"], options["repo"], options["sha"]
            ):
                raise RevisionNotFound(options["sha"], uri) from e
            raise LocationNotFound(
                f"Could not resolve github location {uri}"
            ) from e
        return fs, path

    @classmethod
    def get_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GithubFileSystem,
    ):
        fullpath = posixpath.join(project or "", path)
        return (
            f"https://github.com/{fs.org}/{fs.repo}/tree/{fs.root}/{fullpath}"
        )

    @classmethod
    def pre_process(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ):
        if fs is not None and not isinstance(fs, GithubFileSystem):
            raise TypeError(
                f"{path, project, rev, fs} cannot be resolved by {cls}: fs should be GithubFileSystem, not {fs.__class__}"
            )
        if (
            isinstance(fs, GithubFileSystem)
            and rev is not None
            and fs.root != rev
        ):
            fs.root = rev
            fs.invalidate_cache()

        return path, project, rev, fs

    @classmethod
    def get_project_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GithubFileSystem,
        uri: str,
    ):
        return f"https://github.com/{fs.org}/{fs.repo}/"


class FSSpecResolver(UriResolver):
    """Resolve different fsspec URIs"""

    @classmethod
    def check(
        cls,
        path: str,
        project: Optional[str],
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
        project: Optional[str],
        rev: Optional[str],
        fs: AbstractFileSystem,
    ):
        fullpath = posixpath.join(project or "", path)
        protocol = fs.protocol
        if isinstance(protocol, (tuple, list)):
            if any(fullpath.startswith(p) for p in protocol):
                return fullpath
            protocol = protocol[0]
        if fullpath.startswith(protocol):
            return fullpath
        return f"{protocol}://{fullpath}"


def get_fs(uri: str):
    location = UriResolver.resolve(path=uri, project=None, rev=None, fs=None)
    return location.fs, location.fullpath


def get_path_by_fs_path(fs: AbstractFileSystem, path: str):
    """Restore full uri from fs and path

    Not ideal, but alternative to this is to save uri on MlemObject level and pass it everywhere
    Another alternative is to support this on fsspec level, but we need to contribute it ourselves"""
    return UriResolver.find_resolver(path, None, None, fs=fs).get_uri(
        path, None, None, fs=fs
    )


def get_uri(fs: AbstractFileSystem, path: str, repr: bool = False):
    loc = UriResolver.resolve(path, None, None, fs=fs)
    if repr:
        return loc.uri_repr
    return loc.uri


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
            f"{uri} is not a valid MLEM metafile or a folder with a MLEM model or data"
        )
    raise FileNotFoundError(uri)
