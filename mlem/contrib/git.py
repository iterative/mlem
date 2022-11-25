"""Local git repos support
Extension type: uri

Implementation of `LocalGitResolver`
"""
import os
import posixpath
from typing import ClassVar, Optional, Tuple

from fsspec import AbstractFileSystem, get_fs_token_paths
from fsspec.implementations.git import GitFileSystem
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from mlem.core.meta_io import UriResolver


class LocalGitResolver(UriResolver):
    """Resolve git repositories on local fs"""

    type: ClassVar = "local_git"
    versioning_support: ClassVar = True

    @classmethod
    def check(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: Optional[AbstractFileSystem],
    ) -> bool:
        if isinstance(fs, GitFileSystem):
            return True
        if rev is None:
            return False
        return cls._find_local_git(path) is not None

    @classmethod
    def get_fs(
        cls, uri: str, rev: Optional[str]
    ) -> Tuple[AbstractFileSystem, str]:
        git_dir = cls._find_local_git(uri)
        fs, _, (path,) = get_fs_token_paths(
            os.path.relpath(uri, git_dir),
            protocol="git",
            storage_options={"ref": rev, "path": git_dir},
        )
        return fs, path

    @classmethod
    def get_uri(
        cls,
        path: str,
        project: Optional[str],
        rev: Optional[str],
        fs: GitFileSystem,
    ):
        fullpath = posixpath.join(project or "", path)
        return f"git://{fs.repo.workdir}:{rev or fs.ref}@{fullpath}"

    @classmethod
    def _find_local_git(cls, path: str) -> Optional[str]:
        try:
            return Repo(path, search_parent_directories=True).working_dir
        except (InvalidGitRepositoryError, NoSuchPathError):
            return None
