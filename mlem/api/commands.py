"""
MLEM's Python API
"""
import posixpath
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import click
from fsspec import AbstractFileSystem

from mlem.api.utils import (
    ensure_meta,
    ensure_mlem_object,
    get_dataset_value,
    get_model_meta,
    parse_import_type_modifier,
)
from mlem.config import CONFIG_FILE
from mlem.core.errors import (
    MlemObjectNotFound,
    MlemObjectNotSavedError,
    MlemRootNotFound,
)
from mlem.core.import_objects import ImportAnalyzer
from mlem.core.meta_io import MLEM_DIR, MLEM_EXT, UriResolver, get_fs
from mlem.core.metadata import load_meta, save
from mlem.core.objects import (
    DatasetMeta,
    DeployMeta,
    MlemLink,
    MlemMeta,
    ModelMeta,
    TargetEnvMeta,
)
from mlem.pack import Packager
from mlem.runtime.server.base import Server
from mlem.utils.root import mlem_repo_exists


def apply(
    model: Union[str, ModelMeta],
    *data: Union[str, DatasetMeta, Any],
    method: str = None,
    output: str = None,
    link: bool = False,
) -> Optional[Any]:
    """Apply provided model against provided data

    Args:
        model (ModelMeta): MLEM model.
        data (Any): Input to the model.
        method (str, optional): Which model method to use.
            If None, use the only method model has.
            If more than one is available, will fail.
        output (str, optional): If value is provided,
            assume it's path and save output there.
        link (bool): Whether to create a link to saved output in MLEM root folder.

    Returns:
        If `output=None`, returns results for given data.
            Otherwise returns None.

    # TODO https://github.com/iterative/mlem/issues/25
    # one may want to pass several objects instead of one as `data`
    We may do this by using `*data` or work with `data` being an iterable.
    """
    model = get_model_meta(model)
    w = model.model_type
    res = [
        w.call_method(w.resolve_method(method), get_dataset_value(part))
        for part in data
    ]
    if output is None:
        if len(res) == 1:
            return res[0]
        return res
    if len(res) == 1:
        return save(res[0], output, link=link)

    raise NotImplementedError(
        "Saving several input data objects is not implemented yet"
    )


def clone(
    path: str,
    target: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    target_repo: Optional[str] = None,
    target_fs: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
    link: bool = None,
    external: bool = None,
) -> MlemMeta:
    """Clones MLEM object from `path` to `out`
        and returns Python representation for the created object

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        target (str): Path to save the copy of initial object to.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        fs (Optional[AbstractFileSystem], optional): filesystem to load object from
        target_repo (Optional[str], optional): path to repo to save cloned object to
        target_fs (Optional[AbstractFileSystem], optional): target filesystem
        follow_links (bool, optional): If object we read is a MLEM link, whether to load
            the actual object link points to. Defaults to True.
        load_value (bool, optional): Load actual python object incorporated in MlemMeta object. Defaults to False.
        link: whether to create link in target repo
        external: wheter to put object inside mlem dir in target repo

    Returns:
        MlemMeta: Copy of initial object saved to `out`
    """
    meta = load_meta(
        path,
        repo=repo,
        rev=rev,
        fs=fs,
        follow_links=follow_links,
        load_value=load_value,
    )
    return meta.clone(
        target, fs=target_fs, repo=target_repo, link=link, external=external
    )


def init(path: str = ".") -> None:
    """Creates .mlem directory in `path`"""
    path = posixpath.join(path, MLEM_DIR)
    fs, path = get_fs(path)
    if fs.exists(path):
        click.echo(f"{path} already exists, no need to run `mlem init` again")
    else:
        from mlem import analytics

        if analytics.is_enabled():
            click.echo(
                "MLEM has been initialized.\n"
                "MLEM has anonymous aggregate usage analytics enabled.\n"
                "To opt out set MLEM_NO_ANALYTICS env to 'true' or add 'no_analytics: true' to .mlem/config.yaml\n"
            )
        fs.makedirs(path)
        # some fs dont support creating empty dirs
        with fs.open(posixpath.join(path, CONFIG_FILE), "w"):
            pass


def link(
    source: Union[str, MlemMeta],
    source_repo: Optional[str] = None,
    rev: Optional[str] = None,
    target: Optional[str] = None,
    target_repo: [str] = None,
    external: Optional[bool] = None,
    follow_links: bool = True,
    absolute: bool = False,
) -> MlemLink:
    """Creates MlemLink for an `source` object and dumps it if `target` is provided

    Args:
        source (Union[str, MlemMeta]): The object to create link from.
        source_repo (str, optional): Path to mlem repo where to load obj from
        rev (str, optional): Revision if object is stored in git repo.
        target (str, optional): Where to store the link object.
        target_repo (str, optional): If provided,
            treat `target` as link name and dump link in MLEM DIR
        follow_links (bool): Whether to make link to the underlying object
            if `source` is itself a link. Defaults to True.
        external (bool): Whether to save link outside mlem dir
        absolute (bool): Whether to make link absolute or relative to mlem repo

    Returns:
        MlemLink: Link object to the `source`.
    """
    if isinstance(source, MlemMeta):
        if not source.is_saved:
            raise MlemObjectNotSavedError("Cannot link not saved meta object")
    else:
        source = load_meta(
            source,
            repo=source_repo,
            rev=rev,
            follow_links=follow_links,
        )

    return source.make_link(
        target,
        repo=target_repo,
        external=external,
        absolute=absolute,
    )


def pack(
    packager: Union[str, Packager],
    model: Union[str, ModelMeta],
    out: str,
    **packager_kwargs,
):
    """Pack model in docker-build-ready folder or directly build a docker image.

    Args:
        packager (Union[str, Packager]): Packager to use.
            Out-of-the-box supported string values are "docker_dir" and "docker".
        model (Union[str, ModelMeta]): The model to pack.
        out (str): Path for "docker_dir", image name for "docker".
    """
    model = get_model_meta(model)
    return ensure_mlem_object(Packager, packager, **packager_kwargs).package(
        model, out
    )


def serve(model: ModelMeta, server: Union[Server, str], **server_kwargs):
    """Serve model via HTTP/HTTPS.

    Args:
        model (ModelMeta): The model to serve.
        server (Union[Server, str]): Out-of-the-box supported one is "fastapi".
    """
    from mlem.runtime.interface.base import ModelInterface

    model.load_value()
    interface = ModelInterface(model_type=model.model_type)

    server_obj = ensure_mlem_object(Server, server, **server_kwargs)
    server_obj.serve(interface)


def ls(
    repo: str = ".",
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    type_filter: Union[Type[MlemMeta], Iterable[Type[MlemMeta]], None] = None,
    include_links: bool = True,
) -> Dict[Type[MlemMeta], List[MlemMeta]]:
    if type_filter is None:
        type_filter = set(MlemMeta.non_abstract_subtypes().values())
    if isinstance(type_filter, type) and issubclass(type_filter, MlemMeta):
        type_filter = {type_filter}
    type_filter = set(type_filter)
    if len(type_filter) == 0:
        return {}
    if MlemLink not in type_filter:
        type_filter.add(MlemLink)
    loc = UriResolver.resolve("", repo=repo, rev=rev, fs=fs, find_repo=True)
    if loc.repo is None:
        raise MlemRootNotFound(repo, loc.fs)
    mlem_repo_exists(loc.repo, loc.fs, raise_on_missing=True)
    repo, fs = loc.repo, loc.fs
    res = defaultdict(list)
    for cls in type_filter:
        root_path = posixpath.join(repo, MLEM_DIR, cls.object_type)
        files = fs.glob(
            posixpath.join(root_path, f"**{MLEM_EXT}"), recursive=True
        )
        for file in files:
            meta = load_meta(
                posixpath.relpath(file, repo),
                repo=repo,
                rev=rev,
                follow_links=False,
                fs=fs,
                load_value=False,
            )
            obj_type = cls
            if isinstance(meta, MlemLink):
                link_name = posixpath.relpath(file, root_path)[
                    : -len(MLEM_EXT)
                ]
                is_auto_link = meta.path == link_name + MLEM_EXT

                obj_type = MlemMeta.__type_map__[meta.link_type]
                if obj_type not in type_filter:
                    continue
                if is_auto_link:
                    meta = meta.load_link()
                elif not include_links:
                    continue
            res[obj_type].append(meta)
    return res


def import_object(
    path: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    target: Optional[str] = None,
    target_repo: Optional[str] = None,
    target_fs: Optional[AbstractFileSystem] = None,
    type_: Optional[str] = None,
    copy_data: bool = True,
    external: bool = None,
    link: bool = None,
):
    """Try to load an object as MLEM model (or dataset) and return it,
    optionally saving to the specified target location
    """
    loc = UriResolver.resolve(path, repo, rev, fs)
    if type_ is not None:
        type_, modifier = parse_import_type_modifier(type_)
        if type_ not in ImportAnalyzer.types:
            raise ValueError(f"Unknown import type {type_}")
        meta = ImportAnalyzer.types[type_].process(
            loc, copy_data=copy_data, modifier=modifier
        )
    else:
        meta = ImportAnalyzer.analyze(loc, copy_data=copy_data)
    if target is not None:
        meta.dump(
            target,
            fs=target_fs,
            repo=target_repo,
            link=link,
            external=external,
        )
    return meta


def deploy(
    deploy_meta_or_path: Union[DeployMeta, str],
    model: Union[ModelMeta, str] = None,
    env: Union[TargetEnvMeta, str] = None,
    repo: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    external: bool = None,
    link: bool = None,
    **deploy_kwargs,
) -> DeployMeta:
    if isinstance(deploy_meta_or_path, str):
        try:
            deploy_meta = load_meta(
                path=deploy_meta_or_path,
                repo=repo,
                fs=fs,
                force_type=DeployMeta,
            )
        except MlemObjectNotFound as e:
            if model is None or env is None:
                raise ValueError(
                    "Please provide model and env args for new deployment"
                ) from e
            model_meta = get_model_meta(model)
            env_meta = ensure_meta(TargetEnvMeta, env)
            deploy_meta = env_meta.deploy_type(
                model=model_meta,
                env=env_meta,
                env_link=env_meta.make_link(),
                model_link=model_meta.make_link(),
                **deploy_kwargs,
            )
            deploy_meta.dump(deploy_meta_or_path, fs, repo, link, external)
    else:
        deploy_meta = deploy_meta_or_path

    deploy_meta.deploy()
    return deploy_meta
