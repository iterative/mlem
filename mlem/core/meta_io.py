"""
Utils functions that parse and process supplied URI, serialize/derialize MLEM objects
"""
import contextlib
import os
import posixpath
from abc import ABC, abstractmethod
from inspect import isabstract
from typing import ClassVar, List, Optional, Tuple, Type

from fsspec import AbstractFileSystem, get_fs_token_paths
from fsspec.implementations.local import LocalFileSystem
from pydantic import BaseModel

from mlem.core.base import MlemABC
from mlem.core.errors import (
    HookNotFound,
    InvalidArgumentError,
    LocationNotFound,
    MlemObjectNotFound,
    RevisionNotFound,
)
from mlem.utils.root import find_project_root

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

    @property
    def dirname(self):
        return posixpath.dirname(self.fullpath)

    @property
    def basename(self):
        return posixpath.basename(self.path)

    @contextlib.contextmanager
    def open(self, mode="r", make_dir: bool = False, **kwargs):
        if make_dir:
            self.fs.makedirs(posixpath.dirname(self.fullpath), exist_ok=True)
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
        if os.path.isabs(self.path) and not os.path.isabs(path):
            path = posixpath.join(posixpath.dirname(self.path), path)
        self.uri = self.uri[: -len(self.path)] + path
        self.path = path

    def exists(self):
        return self.fs.exists(self.fullpath)

    def delete(self):
        self.fs.delete(self.fullpath)

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

    @classmethod
    def resolve(
        cls,
        path: str,
        project: str = None,
        rev: str = None,
        fs: AbstractFileSystem = None,
        find_project: bool = False,
    ):
        return UriResolver.resolve(
            path=path,
            project=project,
            rev=rev,
            fs=fs,
            find_project=find_project,
        )


class UriResolver(MlemABC):
    """Base class for resolving location. Turns (path, project, rev, fs) tuple
    into a normalized `Location` instance"""

    abs_name: ClassVar = "resolver"

    class Config:
        type_root = True

    impls: ClassVar[List[Type["UriResolver"]]] = []
    low_priority: ClassVar[bool] = False
    versioning_support: ClassVar[bool] = False

    def __init_subclass__(cls, *args, **kwargs):
        if not isabstract(cls) and cls not in cls.impls:
            if cls.low_priority:
                cls.impls.append(cls)
            else:
                cls.impls.insert(0, cls)
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


class CloudGitResolver(UriResolver, ABC):
    FS: ClassVar[Type[AbstractFileSystem]]
    PROTOCOL: ClassVar[str]
    PREFIXES: ClassVar[List[str]]
    versioning_support: ClassVar = True

    @classmethod
    def check(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> bool:
        fullpath = posixpath.join(project or "", path)
        return isinstance(fs, cls.FS) or any(
            fullpath.startswith(h) for h in cls.PREFIXES
        )

    @classmethod
    def get_fs(
        cls, uri: str, rev: Optional[str]
    ) -> Tuple[AbstractFileSystem, str]:
        options = cls.get_envs()
        if not uri.startswith(cls.PROTOCOL):
            try:
                kwargs = cls.get_kwargs(uri)
            except ValueError as e:
                raise LocationNotFound(*e.args) from e
            options.update(kwargs)
            path = options.pop("path")
            options["sha"] = rev or options.get("sha", None)
        else:
            path = uri
        try:
            fs, _, (path,) = get_fs_token_paths(
                path, protocol=cls.PROTOCOL, storage_options=options
            )
        except FileNotFoundError as e:  # TODO catch HTTPError for wrong orgrepo
            if options["sha"] is not None and not cls.check_rev(options):
                raise RevisionNotFound(options["sha"], uri) from e
            raise LocationNotFound(f"Could not resolve location {uri}") from e
        return fs, path

    @classmethod
    def get_envs(cls):
        return {}

    @classmethod
    def get_kwargs(cls, uri):
        raise NotImplementedError

    @classmethod
    def check_rev(cls, options):
        raise NotImplementedError

    @classmethod
    def pre_process(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ):
        if fs is not None and not isinstance(fs, cls.FS):
            raise TypeError(
                f"{path, project, rev, fs} cannot be resolved by {cls}: fs should be {cls.FS.__class__}, not {fs.__class__}"
            )
        if isinstance(fs, cls.FS) and rev is not None and fs.root != rev:
            fs.root = rev
            fs.invalidate_cache()

        return path, project, rev, fs


class FSSpecResolver(UriResolver):
    """Resolve different fsspec URIs"""

    type: ClassVar = "fsspec"
    low_priority: ClassVar = True

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


def get_fs(uri: str) -> Tuple[AbstractFileSystem, str]:
    location = Location.resolve(path=uri, project=None, rev=None, fs=None)
    return location.fs, location.fullpath


def get_path_by_fs_path(fs: AbstractFileSystem, path: str):
    """Restore full uri from fs and path

    Not ideal, but alternative to this is to save uri on MlemObject level and pass it everywhere
    Another alternative is to support this on fsspec level, but we need to contribute it ourselves"""
    return UriResolver.find_resolver(path, None, None, fs=fs).get_uri(
        path, None, None, fs=fs
    )


def get_uri(fs: AbstractFileSystem, path: str, repr: bool = False):
    loc = Location.resolve(path, None, None, fs=fs)
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

    if fs.isfile(uri + MLEM_EXT):
        # .../name without <MLEM_EXT>
        return uri + MLEM_EXT
    if fs.exists(uri):
        raise MlemObjectNotFound(
            f"{uri} is not a valid MLEM metafile or a folder with a MLEM model or data"
        )
    raise FileNotFoundError(uri)
