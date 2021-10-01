from typing import Any, Optional, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.github import GithubFileSystem
from yaml import safe_load

from mlem.core.meta_io import get_envs, get_fs, get_git_kwargs, get_meta_path
from mlem.core.objects import DatasetMeta, MlemMeta, ModelMeta, find_object
from mlem.utils.root import find_mlem_root


def get_object_metadata(obj: Any, tmp_sample_data=None) -> MlemMeta:
    """Convert given object to appropriate MlemMeta subclass"""
    try:
        return DatasetMeta.from_data(obj)
    except ValueError:  # TODO need separate analysis exception
        return ModelMeta.from_obj(obj, test_data=tmp_sample_data)


def save(
    obj: Any,
    path: str,
    dvc: bool = False,
    tmp_sample_data=None,
    fs: Union[str, AbstractFileSystem] = None,
    link: bool = True,
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
    meta.dump(path, fs=fs, link=link)
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
        path, repo=repo, rev=rev, follow_links=follow_links, load_value=True
    )
    return meta.get_value()


def load_meta(
    path: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
) -> MlemMeta:
    """Load MlemMeta object

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        follow_links (bool, optional): If object we read is a MLEM link, whether to load the actual object link points to. Defaults to True.
        load_value (bool, optional): Load actual python object incorporated in MlemMeta object. Defaults to False.

    Returns:
        MlemMeta: Saved MlemMeta object
    """
    if repo is not None:
        if "github" not in repo:
            raise NotImplementedError("Only Github is supported as of now")
        kwargs = get_envs()

        git_kwargs = get_git_kwargs(repo)
        fs = GithubFileSystem(
            org=git_kwargs["org"], repo=git_kwargs["repo"], sha=rev, **kwargs
        )
    else:
        fs, path = get_fs(path)
    path = find_meta_path(path, fs=fs)
    with fs.open(path, mode="r") as f:
        res = f.read()
    object_type = safe_load(res)["object_type"]
    cls = MlemMeta.subtype_mapping()[object_type]
    meta = cls.read(path, fs=fs, follow_links=follow_links)
    if load_value:
        meta.load_value()
    return meta


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
        mlem_root = find_mlem_root(path=path, fs=fs)
        _, path = find_object(path, fs=fs, mlem_root=mlem_root)

    return path
