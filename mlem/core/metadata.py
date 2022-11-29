"""
Functions to work with metadata: saving, loading,
searching for MLEM object by given path.
"""
import logging
import os
import posixpath
from collections import defaultdict
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, overload

from fsspec import AbstractFileSystem
from typing_extensions import Literal

from mlem.core.data_type import DataType
from mlem.core.errors import (
    HookNotFound,
    MlemObjectNotFound,
    MlemProjectNotFound,
    WrongMetaType,
)
from mlem.core.meta_io import MLEM_EXT, Location, get_meta_path
from mlem.core.model import ModelType
from mlem.core.objects import (
    MlemData,
    MlemLink,
    MlemModel,
    MlemObject,
    find_object,
)
from mlem.telemetry import api_telemetry, telemetry
from mlem.utils.path import make_posix

logger = logging.getLogger(__name__)


def get_object_metadata(
    obj: Any,
    sample_data=None,
    params: Dict[str, str] = None,
) -> Union[MlemData, MlemModel]:
    """Convert given object to appropriate MlemObject subclass"""
    try:
        return MlemData.from_data(
            obj,
            params=params,
        )
    except HookNotFound:
        return MlemModel.from_obj(
            obj,
            sample_data=sample_data,
            params=params,
        )


def log_meta_params(meta: MlemObject, add_object_type: bool = False):
    if add_object_type:
        telemetry.log_param("object_type", meta.object_type)
    if isinstance(meta, MlemModel):
        telemetry.log_param(
            "model_type", meta.model_type_raw[ModelType.__config__.type_field]
        )
    elif isinstance(meta, MlemData):
        data_type = None
        if meta.data_type is not None:
            data_type = meta.data_type.type
        if data_type is None and meta.reader is not None:
            data_type = meta.reader_raw["data_type"][
                DataType.__config__.type_field
            ]
        if data_type is not None:
            telemetry.log_param("data_type", data_type)
    elif meta.__parent__ is not MlemObject:
        telemetry.log_param(f"{meta.object_type}_type", meta.__get_alias__())


@api_telemetry
def save(
    obj: Any,
    path: Union[str, os.PathLike],
    project: Optional[str] = None,
    sample_data=None,
    fs: Optional[AbstractFileSystem] = None,
    params: Dict[str, str] = None,
) -> MlemObject:
    """Saves given object to a given path

    Args:
        obj: Object to dump
        path: If not located on LocalFileSystem, then should be uri
            or `fs` argument should be provided
        project: path to mlem project (optional)
        sample_data: If the object is a model or function, you can
            provide input data sample, so MLEM will include it's schema
            in the model's metadata
        fs: FileSystem for the `path` argument
        params: arbitrary params for object

    Returns:
        None
    """
    meta = get_object_metadata(
        obj,
        sample_data,
        params=params,
    )
    log_meta_params(meta, add_object_type=True)
    path = os.fspath(path)
    meta.dump(path, fs=fs, project=project)
    return meta


def load(
    path: Union[str, os.PathLike],
    project: Optional[str] = None,
    rev: Optional[str] = None,
    batch_size: Optional[int] = None,
    follow_links: bool = True,
) -> Any:
    """Load python object saved by MLEM

    Args:
        path: Path to the object. Could be local path or path inside a git repo.
        project: URL to project if object is located there.
        rev: revision, could be git commit SHA, branch name or tag.
        follow_links: If object we read is a MLEM link, whether to load the
            actual object link points to. Defaults to True.

    Returns:
        Any: Python object saved by MLEM
    """
    path = os.fspath(path)
    meta = load_meta(
        path,
        project=project,
        rev=rev,
        follow_links=follow_links,
        load_value=batch_size is None,
    )
    if isinstance(meta, MlemData) and batch_size:
        return meta.read_batch(batch_size)
    return meta.get_value()


T = TypeVar("T", bound=MlemObject)


@overload
def load_meta(
    path: Union[str, os.PathLike],
    project: Optional[str] = None,
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
    path: Union[str, os.PathLike],
    project: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
    fs: Optional[AbstractFileSystem] = None,
    *,
    force_type: Optional[Type[T]] = None,
) -> T:
    ...


@api_telemetry
def load_meta(
    path: Union[str, os.PathLike],
    project: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
    fs: Optional[AbstractFileSystem] = None,
    *,
    force_type: Optional[Type[T]] = None,
) -> T:
    """Load MlemObject

    Args:
        path: Path to the object. Could be local path or path inside a git repo.
        project: URL to project if object is located there.
        rev: revision, could be git commit SHA, branch name or tag.
        follow_links: If object we read is a MLEM link, whether to load the
            actual object link points to. Defaults to True.
        load_value: Load actual python object incorporated in MlemObject. Defaults to False.
        fs: filesystem to load from. If not provided, will be inferred from path
        force_type: type of meta to be loaded. Defaults to MlemObject (any mlem meta)
    Returns:
        MlemObject: Saved MlemObject
    """
    path = os.fspath(path)
    location = Location.resolve(
        path=make_posix(path),
        project=make_posix(project),
        rev=rev,
        fs=fs,
        find_project=True,
    )
    cls = force_type or MlemObject
    meta = cls.read(
        location=find_meta_location(location),
        follow_links=follow_links,
    )
    log_meta_params(meta, add_object_type=True)
    if load_value:
        meta.load_value()
    if not isinstance(meta, cls):
        raise WrongMetaType(meta, force_type)

    logger.debug("Loaded meta object %s", meta)
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
        path = get_meta_path(uri=location.fullpath, fs=location.fs)
    except FileNotFoundError:
        path = None

    if path is None:
        try:
            _, path = find_object(
                location.path, fs=location.fs, project=location.project
            )
        except (ValueError, MlemProjectNotFound) as e:
            raise MlemObjectNotFound(
                f"MLEM object was not found at `{location.uri_repr}`"
            ) from e
    if location.project is not None:
        path = posixpath.relpath(path, location.project)
    location.update_path(path)
    return location


def list_objects(
    path: str = ".", fs: Optional[AbstractFileSystem] = None, recursive=True
) -> Dict[Type[MlemObject], List[MlemObject]]:
    loc = Location.resolve(path, fs=fs)
    result = defaultdict(list)
    postfix = f"/**{MLEM_EXT}" if recursive else f"/*{MLEM_EXT}"
    for filepath in loc.fs.glob(loc.fullpath + postfix, detail=False):
        meta = load_meta(
            filepath, fs=loc.fs, load_value=False, follow_links=False
        )
        type_ = meta.__class__
        if isinstance(meta, MlemLink):
            type_ = meta.link_cls
        else:
            parent = meta.__parent__
            if (
                parent is not None
                and parent != MlemObject
                and issubclass(parent, MlemObject)
            ):
                type_ = parent
        result[type_].append(meta)
    return result
