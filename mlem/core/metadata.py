"""
Functions to work with metadata: saving, loading,
searching for MLEM object by given path.
"""
import os
from typing import Any, Optional, Tuple, Type, TypeVar, Union, overload

from fsspec import AbstractFileSystem
from fsspec.implementations.github import GithubFileSystem
from typing_extensions import Literal

from mlem.core.errors import HookNotFound
from mlem.core.meta_io import (
    get_fs,
    get_meta_path,
    get_path_by_repo_path_rev,
    path_split_postfix,
)
from mlem.core.objects import DatasetMeta, MlemMeta, ModelMeta, find_object
from mlem.utils.root import find_repo_root


def get_object_metadata(obj: Any, tmp_sample_data=None) -> MlemMeta:
    """Convert given object to appropriate MlemMeta subclass"""
    try:
        return DatasetMeta.from_data(obj)
    except HookNotFound:
        return ModelMeta.from_obj(obj, sample_data=tmp_sample_data)


def save(
    obj: Any,
    path: str,
    repo: Optional[str] = None,
    dvc: bool = False,
    tmp_sample_data=None,
    fs: Union[str, AbstractFileSystem] = None,
    link: bool = True,
    external: Optional[bool] = None,
) -> MlemMeta:
    """Saves given object to a given path

    Args:
        obj: Object to dump
        path: If not located on LocalFileSystem, then should be uri
            or `fs` argument should be provided
        dvc: Store the object's artifacts with dvc
        tmp_sample_data: If the object is a model or function, you can
            provide input data sample, so MLEM will include it's schema
            in the model's metadata
        fs: FileSystem for the `path` argument
        link: Whether to create a link in .mlem folder found for `path`

    Returns:
        None
    """
    meta = get_object_metadata(obj, tmp_sample_data)
    meta.dump(path, fs=fs, repo=repo, link=link, external=external)
    if dvc:
        # TODO dvc add ./%name% https://github.com/iterative/mlem/issues/47
        raise NotImplementedError()
    return meta


def load(
    path: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
) -> Any:
    """Load python object saved by MLEM

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        follow_links (bool, optional): If object we read is a MLEM link, whether to load the actual object link points to. Defaults to True.

    Returns:
        Any: Python object saved by MLEM
    """
    meta = load_meta(
        path,
        repo=repo,
        rev=rev,
        follow_links=follow_links,
        load_value=True,
    )
    return meta.get_value()


T = TypeVar("T", bound=MlemMeta)


@overload
def load_meta(
    path: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
    fs: Optional[AbstractFileSystem] = None,
    *,
    force_type: Literal[None] = None,
) -> MlemMeta:
    ...


def load_meta(
    path: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
    fs: Optional[AbstractFileSystem] = None,
    *,
    force_type: Optional[Type[T]] = None,
) -> T:
    """Load MlemMeta object

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        follow_links (bool, optional): If object we read is a MLEM link, whether to load the actual object link points to. Defaults to True.
        load_value (bool, optional): Load actual python object incorporated in MlemMeta object. Defaults to False.
        fs: filesystem to load from. If not provided, will be inferred from path
        force_type: type of meta to be loaded. Defaults to MlemMeta (any mlem meta)
    Returns:
        MlemMeta: Saved MlemMeta object
    """
    fs, repo, path = get_fs_repo_path_by_path_repo_rev_fs(path, repo, rev, fs)
    meta = MlemMeta.read(path, fs=fs, repo=repo, follow_links=follow_links)
    if load_value:
        meta.load_value()
    if not isinstance(meta, force_type or MlemMeta):
        raise TypeError(
            f"Wrong type of meta loaded, {meta} is not {force_type}"
        )
    return meta  # type: ignore[return-value]


def get_fs_repo_path_by_path_repo_rev_fs(
    path: str,
    repo: Optional[str],
    rev: Optional[str],
    fs: Optional[AbstractFileSystem],
) -> Tuple[AbstractFileSystem, Optional[str], str]:
    if rev is not None:
        if fs is not None:
            if isinstance(fs, GithubFileSystem):
                fs = GithubFileSystem(
                    fs.org, fs.repo, rev, fs.username, fs.token
                )
            else:
                raise NotImplementedError(
                    f"rev is supported only for github, not {fs}"
                )
        else:
            if repo is not None:
                gh_path, fs_kwargs = get_path_by_repo_path_rev(repo, path, rev)
            else:
                gh_path, fs_kwargs = path, {"sha": rev}
            fs, repo_path = get_fs(gh_path, **fs_kwargs)
            if not isinstance(fs, GithubFileSystem):
                raise NotImplementedError(
                    f"rev is supported only for github, not {fs}"
                )
            if repo is not None:
                repo = path_split_postfix(repo_path, path)
            path = repo_path
    if fs is None:
        fs, path = get_fs(os.path.join(repo or "", path))
    fullpath = find_meta_path(path, fs=fs)
    return fs, repo, os.path.relpath(fullpath, repo or "")


def find_meta_path(path: str, fs: AbstractFileSystem) -> str:
    """Locate MlemMeta object by given path

    Args:
        path (str): Path to object or a link name.
        fs (AbstractFileSystem): Filesystem for the given path

    Returns:
        str: Path to located object

    TODO https://github.com/iterative/mlem/issues/4
    We should do all the following types of addressing to object
    * data/model # by original binary (good for autocomplition)
    * .mlem/models/data/model.mlem.yaml # by path to metafile in mlem
    * meta/aaa/bbb/mymetafile.yaml # by path to metafile not in
    * http://example.com/file.mlem.yaml # remote metafile
    * git+.... # remote git
    """
    try:
        # first, assume `path` is the filename
        # this allows to find the object not listed in .mlem/
        path = get_meta_path(uri=path, fs=fs)
    except FileNotFoundError:
        # now search for objects in .mlem
        # TODO: exceptions thrown here doesn't explain that
        #  direct search by path was also failed. Need to clarify
        repo = find_repo_root(path=path, fs=fs)
        _, path = find_object(path, fs=fs, repo=repo)

    return path
