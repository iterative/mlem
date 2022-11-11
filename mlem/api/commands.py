"""
MLEM's Python API
"""
import posixpath
from typing import Any, Dict, Optional, Union

from fsspec import AbstractFileSystem

from mlem.api.utils import (
    ensure_meta,
    ensure_mlem_object,
    get_data_value,
    get_model_meta,
    parse_import_type_modifier,
)
from mlem.constants import MLEM_CONFIG_FILE_NAME, PREDICT_METHOD_NAME
from mlem.core.errors import (
    InvalidArgumentError,
    MlemError,
    MlemObjectNotFound,
    MlemObjectNotSavedError,
    WrongMethodError,
)
from mlem.core.import_objects import ImportAnalyzer, ImportHook
from mlem.core.meta_io import Location, get_fs
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
from mlem.runtime.interface import prepare_model_interface
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


def apply(
    model: Union[str, MlemModel, Any],
    *data: Union[str, MlemData, Any],
    method: str = None,
    output: str = None,
    target_project: str = None,
    batch_size: Optional[int] = None,
) -> Optional[Any]:
    """Apply provided model against provided data

    Args:
        model: MLEM model.
        data: Input to the model.
        method: Which model method to use.
            If None, use the only method model has.
            If more than one is available, will fail.
        output: If value is provided,
            assume it's path and save output there.
        target_project: Path to MLEM project to save the result to.
        batch_size: If provided, will process data in batches of given size.

    Returns:
        If `output=None`, returns results for given data.
            Otherwise returns None.
    """
    if isinstance(model, (str, MlemModel)):
        model = get_model_meta(model)
    else:
        model = MlemModel.from_obj(model)
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
    return save(res, output, project=target_project)


def apply_remote(
    client: Union[str, Client],
    *data: Union[str, MlemData, Any],
    method: str = None,
    output: str = None,
    target_project: str = None,
    **client_kwargs,
) -> Optional[Any]:
    """Apply provided model against provided data

    Args:
        client: The client to access methods of deployed model.
        data: Input to the model.
        method: Which model method to use.
            If None, use the only method model has.
            If more than one is available, will fail.
        output: If value is provided,
            assume it's path and save output there.
        target_project: Path to MLEM project to save the result to.
        **client_kwargs: Additional arguments to pass to client.

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
    return save(res, output, project=target_project)


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
) -> MlemObject:
    """Clones MLEM object from `path` to `out`
        and returns Python representation for the created object

    Args:
        path: Path to the object. Could be local path or path inside a git repo.
        target: Path to save the copy of initial object to.
        project: URL to project if object is located there.
        rev: revision, could be git commit SHA, branch name or tag.
        fs: filesystem to load object from
        target_project: path to project to save cloned object to
        target_fs: target filesystem
        follow_links: If object we read is a MLEM link, whether to load
            the actual object link points to. Defaults to True.
        load_value: Load actual python object incorporated in MlemObject. Defaults to False.

    Returns:
        MlemObject: Copy of initial object saved to `out`.
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
    )


def init(path: str = ".") -> None:
    """Creates MLEM config in `path`

    Args:
        path: Path to create config in. Defaults to current directory.

    Returns:
        None
    """
    path = posixpath.join(path, MLEM_CONFIG_FILE_NAME)
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
        # some fs dont support creating empty dirs
        with fs.open(path, "w"):
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
    follow_links: bool = True,
    absolute: bool = False,
) -> MlemLink:
    """Creates MlemLink for an `source` object and dumps it if `target` is provided

    Args:
        source: The object to create link from.
        source_project: Path to mlem project where to load obj from
        rev: Revision if object is stored in git repo.
        target: Where to store the link object.
        target_project: If provided,
            treat `target` as link name and dump link in MLEM DIR
        follow_links: Whether to make link to the underlying object
            if `source` is itself a link. Defaults to True.
        absolute: Whether to make link absolute or relative to mlem project

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
        absolute=absolute,
    )


def build(
    builder: Union[str, MlemBuilder],
    model: Union[str, MlemModel],
    **builder_kwargs,
):
    """Pack model into something useful, such as docker image, Python package or something else.

    Args:
        builder: Builder to use.
        model: The model to build.
        builder_kwargs: Additional keyword arguments to pass to the builder.

    Returns:
        The result of the build, different for different builders.
    """
    model = get_model_meta(model, load_value=False)
    return ensure_mlem_object(MlemBuilder, builder, **builder_kwargs).build(
        model
    )


def serve(
    model: Union[str, MlemModel],
    server: Union[Server, str],
    **server_kwargs,
):
    """Serve a model by exposing its methods as endpoints.

    Args:
        model: The model to serve.
        server: Out-of-the-box supported one is "fastapi".
        server_kwargs: Additional kwargs to pass to the server.

    Returns:
        None
    """
    model = get_model_meta(model, load_value=True)

    server_obj = ensure_mlem_object(Server, server, **server_kwargs)
    interface = prepare_model_interface(model, server_obj)
    echo(f"Starting {server_obj.type} server...")
    server_obj.serve(interface)


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
):
    """Try to load an object as MLEM model (or data) and return it,
    optionally saving to the specified target location

    Args:
        path: Path to the object to import.
        project: Path to mlem project where to load obj from.
        rev: Revision if object is stored in git repo.
        fs: Filesystem to use to load the object.
        target: Where to store the imported object.
        target_project: If provided, treat `target` as object name and dump
            object in this MLEM Project.
        target_fs: Filesystem to use to save the object.
        type_: Type of the object to import. If not provided, will try to
            infer from the object itself.
        copy_data: Whether to copy data to the target location.

    Returns:
        MlemObject: Imported object.
    """
    loc = Location.resolve(path, project, rev, fs)
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
        )
    return meta


def deploy(
    deploy_meta_or_path: Union[MlemDeployment, str],
    model: Union[MlemModel, str],
    env: Union[MlemEnv, str] = None,
    project: Optional[str] = None,
    rev: Optional[str] = None,
    fs: Optional[AbstractFileSystem] = None,
    env_kwargs: Dict[str, Any] = None,
    **deploy_kwargs,
) -> MlemDeployment:
    """Deploy a model to a target environment. Can use an existing deployment
    declaration or create a new one on-the-fly.

    Args:
        deploy_meta_or_path: MlemDeployment object or path to it.
        model: The model to deploy.
        env: The environment to deploy to.
        project: Path to mlem project where to load obj from.
        rev: Revision if object is stored in git repo.
        fs: Filesystem to use to load the object.
        env_kwargs: Additional kwargs to pass to the environment.
        deploy_kwargs: Additional kwargs to pass to the deployment.

    Returns:
        MlemDeployment: The deployment object.
    """
    deploy_meta: MlemDeployment
    update = False
    if isinstance(deploy_meta_or_path, str):
        try:
            deploy_meta = load_meta(
                path=deploy_meta_or_path,
                project=project,
                rev=rev,
                fs=fs,
                force_type=MlemDeployment,
            )
            update = True
        except MlemObjectNotFound as e:
            if env is None:
                raise MlemError(
                    "Please provide model and env args for new deployment"
                ) from e
            if not deploy_meta_or_path:
                raise MlemError("deploy_path cannot be empty") from e

            env_meta = ensure_meta(MlemEnv, env, allow_typename=True)
            if isinstance(env_meta, type):
                env = None
                if env_kwargs:
                    env = env_meta(**env_kwargs)
            deploy_type = env_meta.deploy_type
            deploy_meta = deploy_type(
                env=env,
                **deploy_kwargs,
            )
            deploy_meta.dump(deploy_meta_or_path, fs, project)
    else:
        deploy_meta = deploy_meta_or_path
        update = True

    if update:
        pass  # todo update from deploy_args and env_args
    # ensuring links are working
    deploy_meta.get_env()
    model_meta = get_model_meta(model)

    deploy_meta.check_unchanged()
    deploy_meta.deploy(model_meta)
    return deploy_meta
