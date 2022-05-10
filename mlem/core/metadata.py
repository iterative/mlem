"""
Functions to work with metadata: saving, loading,
searching for MLEM object by given path.
"""
import logging
import posixpath
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, overload

from fsspec import AbstractFileSystem
from typing_extensions import Literal

from mlem.core.errors import (
    HookNotFound,
    MlemObjectNotFound,
    MlemRootNotFound,
    WrongMetaType,
)
from mlem.core.meta_io import Location, UriResolver, get_meta_path
from mlem.core.objects import MlemDataset, MlemModel, MlemObject, find_object
from mlem.utils.path import make_posix

logger = logging.getLogger(__name__)


def get_object_metadata(
    obj: Any,
    sample_data=None,
    description: str = None,
    params: Dict[str, str] = None,
    tags: List[str] = None,
) -> Union[MlemDataset, MlemModel]:
    """Convert given object to appropriate MlemObject subclass"""
    try:
        return MlemDataset.from_data(
            obj, description=description, params=params, tags=tags
        )
    except HookNotFound:
        return MlemModel.from_obj(
            obj,
            sample_data=sample_data,
            description=description,
            params=params,
            tags=tags,
        )


def save(
    obj: Any,
    path: str,
    repo: Optional[str] = None,
    sample_data=None,
    fs: Union[str, AbstractFileSystem] = None,
    index: bool = None,
    external: Optional[bool] = None,
    description: str = None,
    params: Dict[str, str] = None,
    tags: List[str] = None,
    update: bool = False,
) -> MlemObject:
    """Saves given object to a given path

    Args:
        obj: Object to dump
        path: If not located on LocalFileSystem, then should be uri
            or `fs` argument should be provided
        repo: path to mlem repo (optional)
        sample_data: If the object is a model or function, you can
            provide input data sample, so MLEM will include it's schema
            in the model's metadata
        fs: FileSystem for the `path` argument
        index: Whether to add object to mlem repo index
        external: if obj is saved to repo, whether to put it outside of .mlem dir
        description: description for object
        params: arbitrary params for object
        tags: tags for object
        update: whether to keep old description/tags/params if new values were not provided

    Returns:
        None
    """
    if update and (description is None or params is None or tags is None):
        try:
            old_meta = load_meta(path, repo=repo, fs=fs, load_value=False)
            description = description or old_meta.description
            params = params or old_meta.params
            tags = tags or old_meta.tags
        except MlemObjectNotFound:
            logger.warning(
                "Saving with update=True, but no existing object found at %s %s %s",
                repo,
                path,
                fs,
            )
    meta = get_object_metadata(
        obj, sample_data, description=description, params=params, tags=tags
    )
    meta.dump(path, fs=fs, repo=repo, index=index, external=external)
    return meta


def load(
    path: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    batch_size: Optional[int] = None,
    follow_links: bool = True,
) -> Any:
    """Load python object saved by MLEM

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        follow_links (bool, optional): If object we read is a MLEM link, whether to load the
            actual object link points to. Defaults to True.

    Returns:
        Any: Python object saved by MLEM
    """
    meta = load_meta(
        path,
        repo=repo,
        rev=rev,
        follow_links=follow_links,
        load_value=batch_size is None,
    )
    if isinstance(meta, MlemDataset) and batch_size:
        return meta.read_batch(batch_size)
    return meta.get_value()


T = TypeVar("T", bound=MlemObject)


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
) -> MlemObject:
    ...


@overload
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
    """Load MlemObject

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        follow_links (bool, optional): If object we read is a MLEM link, whether to load the
            actual object link points to. Defaults to True.
        load_value (bool, optional): Load actual python object incorporated in MlemObject. Defaults to False.
        fs: filesystem to load from. If not provided, will be inferred from path
        force_type: type of meta to be loaded. Defaults to MlemObject (any mlem meta)
    Returns:
        MlemObject: Saved MlemObject
    """
    location = UriResolver.resolve(
        path=make_posix(path),
        repo=make_posix(repo),
        rev=rev,
        fs=fs,
        find_repo=True,
    )
    cls = force_type or MlemObject
    meta = cls.read(
        location=find_meta_location(location),
        follow_links=follow_links,
    )
    if load_value:
        meta.load_value()
    if not isinstance(meta, cls):
        raise WrongMetaType(meta, force_type)
    return meta  # type: ignore[return-value]


def find_meta_location(location: Location) -> Location:
    """Locate MlemObject by given location

    Args:
        location: location to find meta

    Returns:
        location: Resolved metadata file location
    """
    location = location.copy()
    try:
        # first, assume `location` points to an external mlem object
        # this allows to find the object not listed in .mlem/
        path = get_meta_path(uri=location.fullpath, fs=location.fs)
    except FileNotFoundError:
        # now search for objects in .mlem
        try:
            _, path = find_object(
                location.path, fs=location.fs, repo=location.repo
            )
        except (ValueError, MlemRootNotFound) as e:
            raise MlemObjectNotFound(
                f"MLEM object was not found at `{location.uri_repr}`"
            ) from e
    if location.repo is not None:
        path = posixpath.relpath(path, location.repo)
    location.update_path(path)
    return location
