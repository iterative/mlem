import os
from typing import Any, Optional, Union

import click
from pydantic import parse_obj_as

from mlem.core.errors import InvalidArgumentError
from mlem.core.meta_io import MLEM_DIR, deserialize
from mlem.core.metadata import load, load_meta, save
from mlem.core.objects import DatasetMeta, MlemLink, MlemMeta, ModelMeta
from mlem.pack import Packager
from mlem.runtime.server.base import Server


def _get_dataset(dataset: Any) -> Any:
    if isinstance(dataset, str):
        return load(dataset)
    elif isinstance(dataset, DatasetMeta):
        # TODO: https://github.com/iterative/mlem/issues/29
        #  fix discrepancies between model and data meta objects
        if not hasattr(dataset.dataset, "data"):
            dataset.load_value()
        return dataset.data
    else:
        # TODO: https://github.com/iterative/mlem/issues/29
        #  should we check whether this dataset is parseable by MLEM?
        #  I guess not cause one may have a model with data input of unknown format/type
        return dataset


def _get_model_meta(model: Any) -> ModelMeta:
    print(model, type(model))
    if isinstance(model, ModelMeta):
        if model.get_value() is None:
            model.load_value()
        return model
    elif isinstance(model, str):
        model = load_meta(model)
        if not isinstance(model, ModelMeta):
            raise InvalidArgumentError(
                "MLEM object is loaded, but it's not a model as expected"
            )
        model.load_value()
        return model
    else:
        raise InvalidArgumentError(
            f"The object {model} is neither ModelMeta nor path to it"
        )


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
    model = _get_model_meta(model)
    w = model.model
    res = [
        w.call_method(w.resolve_method(method), _get_dataset(part))
        for part in data
    ]
    if output is None:
        if len(res) == 1:
            return res[0]
        else:
            return res
    if len(res) == 1:
        return save(res[0], output, link=link)
    else:
        raise NotImplementedError(
            "Saving several input data objects is not implemented yet"
        )


def get(
    path: str,
    out: str,
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
) -> MlemMeta:
    """Clones MLEM object from `path` to `out`
        and returns Python representation for the created object

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        out (str): Path to save the copy of initial object to.
        repo (Optional[str], optional): URL to repo if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        follow_links (bool, optional): If object we read is a MLEM link, whether to load the actual object link points to. Defaults to True.
        load_value (bool, optional): Load actual python object incorporated in MlemMeta object. Defaults to False.

    Returns:
        MlemMeta: Copy of initial object saved to `out`
    """
    meta = load_meta(
        path,
        repo=repo,
        rev=rev,
        follow_links=follow_links,
        load_value=load_value,
    )
    return meta.clone(out)


def init(path: str = ".") -> None:
    """Creates .mlem directory in `path`"""
    path = os.path.join(path, MLEM_DIR)
    if os.path.exists(path):
        click.echo(f"{path} already exists, no need to run `mlem init` again")
    else:
        os.makedirs(path)


def link(
    source: Union[str, MlemMeta],
    repo: Optional[str] = None,
    rev: Optional[str] = None,
    target: Optional[str] = None,
    mlem_root: Optional[str] = ".",
    follow_links: bool = True,
    check_extension: bool = True,
    absolute: bool = False,
) -> MlemLink:
    """Creates MlemLink for an `source` object and dumps it if `target` is provided

    Args:
        source (Union[str, MlemMeta]): The object to create link from.
        repo (str, optional): Repo if object is stored in git repo.
        rev (str, optional): Revision if object is stored in git repo.
        target (str, optional): Where to store the link object.
        mlem_root (str, optional): If provided,
            treat `target` as link name and dump link in MLEM DIR
        follow_links (bool): Whether to make link to the underlying object
            if `source` is itself a link. Defaults to True.
        check_extension (bool): Whether to check if `target` ends
            with MLEM file extenstion. Defaults to True.

    Returns:
        MlemLink: Link object to the `source`.
    """
    # TODO: https://github.com/iterative/mlem/issues/12
    # right now this only works when source and target are on local FS
    # (we don't even throw an NotImplementedError yet)
    # need to support other cases, like `source="github://..."`
    if repo is not None:
        raise NotImplementedError()
    if isinstance(source, MlemMeta):
        if source.name is None:
            raise ValueError("Cannot link not saved meta object")
    else:
        source = load_meta(source, follow_links=follow_links)

    link = source.make_link()
    if target is not None:
        link.dump(
            target,
            mlem_root=mlem_root,
            check_extension=check_extension,
            absolute=absolute,
        )
    return link


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
    if isinstance(model, str):
        meta = load_meta(model)
        if not isinstance(meta, ModelMeta):
            raise ValueError(f"{model} is not a model")
        model = meta
    if isinstance(packager, str):
        packager = deserialize({"type": packager, **packager_kwargs}, Packager)
    packager.package(model, out)


def serve(model: ModelMeta, server: Union[Server, str], **server_kwargs):
    """Serve model via HTTP/HTTPS.

    Args:
        model (ModelMeta): The model to serve.
        server (Union[Server, str]): Out-of-the-box supported one is "fastapi".
    """
    from mlem.runtime.interface.base import ModelInterface

    model.load_value()
    interface = ModelInterface()
    interface.model = model.model

    if not isinstance(server, Server):
        server_obj = parse_obj_as(
            Server, {Server.__type_field__: server, **server_kwargs}
        )
    else:
        server_obj = server
    server_obj.serve(interface)
