"""
MLEM's Python API
"""
import posixpath
from typing import Any, Dict, Iterable, List, Optional, Type, Union

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from mlem.api.utils import (
    ensure_meta,
    ensure_mlem_object,
    get_data_value,
    get_model_meta,
    parse_import_type_modifier,
)
from mlem.config import CONFIG_FILE_NAME, project_config
from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.errors import (
    InvalidArgumentError,
    MlemError,
    MlemObjectNotFound,
    MlemObjectNotSavedError,
    MlemProjectNotFound,
    WrongMethodError,
)
from mlem.core.import_objects import ImportAnalyzer, ImportHook
from mlem.core.meta_io import MLEM_DIR, Location, UriResolver, get_fs
from mlem.core.metadata import load_meta, save
from mlem.core.objects import (
    MlemBuilder,
    MlemData,
    MlemDeployment,
    MlemEnv,
    MlemLink,
    MlemModel,
    MlemObject,
)
from mlem.runtime.client import Client
from mlem.runtime.server import Server
from mlem.ui import (
    EMOJI_APPLY,
    EMOJI_COPY,
    EMOJI_LOAD,
    EMOJI_MLEM,
    boxify,
    color,
    echo,
)
from mlem.utils.root import find_project_root, mlem_project_exists


def apply(
    model: Union[str, MlemModel],
    *data: Union[str, MlemData, Any],
    method: str = None,
    output: str = None,
    target_project: str = None,
    index: bool = None,
    external: bool = None,
    batch_size: Optional[int] = None,
) -> Optional[Any]:
    """Apply provided model against provided data

    Args:
        model (MlemModel): MLEM model.
        data (Any): Input to the model.
        method (str, optional): Which model method to use.
            If None, use the only method model has.
            If more than one is available, will fail.
        output (str, optional): If value is provided,
            assume it's path and save output there.
        index (bool): Whether to index saved output in MLEM root folder.
        external (bool): Whether to save result outside mlem dir

    Returns:
        If `output=None`, returns results for given data.
            Otherwise returns None.

    """
    model = get_model_meta(model)
    w = model.model_type
    try:
        resolved_method = w.resolve_method(method)
    except WrongMethodError:
        resolved_method = PREDICT_METHOD_NAME
    echo(EMOJI_APPLY + f"Applying `{resolved_method}` method...")
    if batch_size:
        res: Any = []
        for part in data:
            batch_data = get_data_value(part, batch_size)
            for batch in batch_data:
                preds = w.call_method(resolved_method, batch.data)
                res += [*preds]  # TODO: merge results
    else:
        res = [
            w.call_method(resolved_method, get_data_value(part))
            for part in data
        ]
    if output is None:
        if len(res) == 1:
            return res[0]
        return res
    if len(res) == 1:
        res = res[0]
    return save(
        res, output, project=target_project, external=external, index=index
    )


def apply_remote(
    client: Union[str, Client],
    *data: Union[str, MlemData, Any],
    method: str = None,
    output: str = None,
    target_project: str = None,
    index: bool = False,
    **client_kwargs,
) -> Optional[Any]:
    """Apply provided model against provided data

    Args:
        client (Client): The client to access methods of deployed model.
        data (Any): Input to the model.
        method (str, optional): Which model method to use.
            If None, use the only method model has.
            If more than one is available, will fail.
        output (str, optional): If value is provided,
            assume it's path and save output there.
        index (bool): Whether to index saved output in MLEM root folder.

    Returns:
        If `output=None`, returns results for given data.
            Otherwise returns None.

    """
    client = ensure_mlem_object(Client, client, **client_kwargs)
    if method is not None:
        try:
            resolved_method = getattr(client, method)
        except WrongMethodError:
            resolved_method = getattr(client, PREDICT_METHOD_NAME)
    else:
        raise InvalidArgumentError("method cannot be None")

    echo(EMOJI_APPLY + f"Applying `{resolved_method.method.name}` method...")
    res = [resolved_method(get_data_value(part)) for part in data]
    if output is None:
        if len(res) == 1:
            return res[0]
        return res
    if len(res) == 1:
        res = res[0]
    return save(res, output, project=target_project, index=index)


def clone(
    path: str,
    target: str,
    project: Optional[str] = None,
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    target_project: Optional[str] = None,
    target_fs: Optional[str] = None,
    follow_links: bool = True,
    load_value: bool = False,
    index: bool = None,
    external: bool = None,
) -> MlemObject:
    """Clones MLEM object from `path` to `out`
        and returns Python representation for the created object

    Args:
        path (str): Path to the object. Could be local path or path inside a git repo.
        target (str): Path to save the copy of initial object to.
        project (Optional[str], optional): URL to project if object is located there.
        rev (Optional[str], optional): revision, could be git commit SHA, branch name or tag.
        fs (Optional[AbstractFileSystem], optional): filesystem to load object from
        target_project (Optional[str], optional): path to project to save cloned object to
        target_fs (Optional[AbstractFileSystem], optional): target filesystem
        follow_links (bool, optional): If object we read is a MLEM link, whether to load
            the actual object link points to. Defaults to True.
        load_value (bool, optional): Load actual python object incorporated in MlemObject. Defaults to False.
        index: whether to index object in target project
        external: wheter to put object inside mlem dir in target project

    Returns:
        MlemObject: Copy of initial object saved to `out`
    """
    meta = load_meta(
        path,
        project=project,
        rev=rev,
        fs=fs,
        follow_links=follow_links,
        load_value=load_value,
    )
    echo(EMOJI_COPY + f"Cloning {meta.loc.uri_repr}")
    if target in ("", "."):
        target = posixpath.basename(meta.loc.uri)
    return meta.clone(
        target,
        fs=target_fs,
        project=target_project,
        index=index,
        external=external,
    )


def init(path: str = ".") -> None:
    """Creates .mlem directory in `path`"""
    path = posixpath.join(path, MLEM_DIR)
    fs, path = get_fs(path)
    if fs.exists(path):
        echo(
            f"{posixpath.abspath(path)} already exists, no need to run `mlem init` again"
        )
    else:
        from mlem.telemetry import telemetry

        echo(
            color("███╗   ███╗", "#13ADC7")
            + color("██╗     ", "#945DD5")
            + color("███████╗", "#F46737")
            + color("███╗   ███╗\n", "#7B61FF")
            + color("████╗ ████║", "#13ADC7")
            + color("██║     ", "#945DD5")
            + color("██╔════╝", "#F46737")
            + color("████╗ ████║\n", "#7B61FF")
            + color("██╔████╔██║", "#13ADC7")
            + color("██║     ", "#945DD5")
            + color("█████╗  ", "#F46737")
            + color("██╔████╔██║\n", "#7B61FF")
            + color("██║╚██╔╝██║", "#13ADC7")
            + color("██║     ", "#945DD5")
            + color("██╔══╝  ", "#F46737")
            + color("██║╚██╔╝██║\n", "#7B61FF")
            + color("██║ ╚═╝ ██║", "#13ADC7")
            + color("███████╗", "#945DD5")
            + color("███████╗", "#F46737")
            + color("██║ ╚═╝ ██║\n", "#7B61FF")
            + color("╚═╝     ╚═╝", "#13ADC7")
            + color("╚══════╝", "#945DD5")
            + color("╚══════╝", "#F46737")
            + color("╚═╝     ╚═╝\n", "#7B61FF")
        )
        if telemetry.is_enabled():
            echo(
                boxify(
                    "MLEM has enabled anonymous aggregate usage analytics.\n"
                    "Read the analytics documentation (and how to opt-out) here:\n"
                    "<https://mlem.ai/docs/user-guide/analytics>"
                )
            )
        fs.makedirs(path)
        # some fs dont support creating empty dirs
        with fs.open(posixpath.join(path, CONFIG_FILE_NAME), "w"):
            pass
        echo(
            EMOJI_MLEM
            + color("What's next?\n---------------", "yellow")
            + """
- Check out the documentation: <https://mlem.ai/docs>
- Star us on GitHub: <https://github.com/iterative/mlem>
- Get help and share ideas: <https://mlem.ai/chat>
"""
        )


def link(
    source: Union[str, MlemObject],
    source_project: Optional[str] = None,
    rev: Optional[str] = None,
    target: Optional[str] = None,
    target_project: Optional[str] = None,
    external: Optional[bool] = None,
    follow_links: bool = True,
    absolute: bool = False,
) -> MlemLink:
    """Creates MlemLink for an `source` object and dumps it if `target` is provided

    Args:
        source (Union[str, MlemObject]): The object to create link from.
        source_project (str, optional): Path to mlem project where to load obj from
        rev (str, optional): Revision if object is stored in git repo.
        target (str, optional): Where to store the link object.
        target_project (str, optional): If provided,
            treat `target` as link name and dump link in MLEM DIR
        follow_links (bool): Whether to make link to the underlying object
            if `source` is itself a link. Defaults to True.
        external (bool): Whether to save link outside mlem dir
        absolute (bool): Whether to make link absolute or relative to mlem project

    Returns:
        MlemLink: Link object to the `source`.
    """
    if isinstance(source, MlemObject):
        if not source.is_saved:
            raise MlemObjectNotSavedError("Cannot link not saved meta object")
    else:
        source = load_meta(
            source,
            project=source_project,
            rev=rev,
            follow_links=follow_links,
        )

    return source.make_link(
        target,
        project=target_project,
        external=external,
        absolute=absolute,
    )


def build(
    builder: Union[str, MlemBuilder],
    model: Union[str, MlemModel],
    **builder_kwargs,
):
    """Pack model in docker-build-ready folder or directly build a docker image.

    Args:
        builder (Union[str, MlemBuilder]): Packager to use.
            Out-of-the-box supported string values are "docker_dir" and "docker".
        model (Union[str, MlemModel]): The model to build.
    """
    model = get_model_meta(model)
    return ensure_mlem_object(MlemBuilder, builder, **builder_kwargs).build(
        model
    )


def serve(model: MlemModel, server: Union[Server, str], **server_kwargs):
    """Serve model via HTTP/HTTPS.

    Args:
        model (MlemModel): The model to serve.
        server (Union[Server, str]): Out-of-the-box supported one is "fastapi".
    """
    from mlem.runtime.interface import ModelInterface

    model.load_value()
    interface = ModelInterface(model_type=model.model_type)

    server_obj = ensure_mlem_object(Server, server, **server_kwargs)
    echo(f"Starting {server_obj.type} server...")
    server_obj.serve(interface)


def _validate_ls_project(loc: Location, project):
    if loc.project is None:
        raise MlemProjectNotFound(project, loc.fs)
    if isinstance(loc.fs, LocalFileSystem):
        loc.project = find_project_root(loc.project, loc.fs)
    else:
        mlem_project_exists(loc.project, loc.fs, raise_on_missing=True)


def ls(  # pylint: disable=too-many-locals
    project: str = ".",
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    type_filter: Union[
        Type[MlemObject], Iterable[Type[MlemObject]], None
    ] = None,
    include_links: bool = True,
    ignore_errors: bool = False,
) -> Dict[Type[MlemObject], List[MlemObject]]:
    loc = UriResolver.resolve(
        "", project=project, rev=rev, fs=fs, find_project=True
    )
    _validate_ls_project(loc, project)
    return project_config(project, fs).index.list(
        loc, type_filter, include_links, ignore_errors
    )


def import_object(
    path: str,
    project: Optional[str] = None,
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    target: Optional[str] = None,
    target_project: Optional[str] = None,
    target_fs: Optional[AbstractFileSystem] = None,
    type_: Optional[str] = None,
    copy_data: bool = True,
    external: bool = None,
    index: bool = None,
):
    """Try to load an object as MLEM model (or data) and return it,
    optionally saving to the specified target location
    """
    loc = UriResolver.resolve(path, project, rev, fs)
    echo(EMOJI_LOAD + f"Importing object from {loc.uri_repr}")
    if type_ is not None:
        type_, modifier = parse_import_type_modifier(type_)
        meta = ImportHook.load_type(type_).process(
            loc, copy_data=copy_data, modifier=modifier
        )
    else:
        meta = ImportAnalyzer.analyze(loc, copy_data=copy_data)
    if target is not None:
        meta.dump(
            target,
            fs=target_fs,
            project=target_project,
            index=index,
            external=external,
        )
    return meta


def deploy(
    deploy_meta_or_path: Union[MlemDeployment, str],
    model: Union[MlemModel, str] = None,
    env: Union[MlemEnv, str] = None,
    project: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    external: bool = None,
    index: bool = None,
    **deploy_kwargs,
) -> MlemDeployment:
    deploy_path = None
    if isinstance(deploy_meta_or_path, str):
        deploy_path = deploy_meta_or_path
        try:
            deploy_meta = load_meta(
                path=deploy_path,
                project=project,
                fs=fs,
                force_type=MlemDeployment,
            )
        except MlemObjectNotFound:
            deploy_meta = None

    else:
        deploy_meta = deploy_meta_or_path
        if model is not None:
            deploy_meta.replace_model(get_model_meta(model))

    if deploy_meta is None:
        if model is None or env is None:
            raise MlemError(
                "Please provide model and env args for new deployment"
            )
        if not deploy_path:
            raise MlemError("deploy_path cannot be empty")
        model_meta = get_model_meta(model)
        env_meta = ensure_meta(MlemEnv, env)
        deploy_meta = env_meta.deploy_type(
            model=model_meta,
            env=env_meta,
            env_link=env_meta.make_link(),
            model_link=model_meta.make_link(),
            **deploy_kwargs,
        )
        deploy_meta.dump(deploy_path, fs, project, index, external)
    # ensuring links are working
    deploy_meta.get_env()
    deploy_meta.get_model()

    deploy_meta.run()
    return deploy_meta
