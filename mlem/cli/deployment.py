import logging
from json import dumps
from typing import List, Optional

import click
from typer import Argument, Option, Typer
from typer.main import get_click_param
from typer.models import ParamMeta

from mlem.cli.apply import run_apply_remote
from mlem.cli.declare import add_env_params_deployment, process_fields
from mlem.cli.main import (
    app,
    mlem_command,
    mlem_group,
    mlem_group_callback,
    option_data_project,
    option_data_rev,
    option_file_conf,
    option_json,
    option_load,
    option_method,
    option_model,
    option_model_project,
    option_model_rev,
    option_project,
    option_rev,
    option_target_project,
)
from mlem.cli.utils import (
    for_each_impl,
    lazy_class_docstring,
    make_not_required,
    wrap_build_error,
)
from mlem.core.base import build_mlem_object
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import DeploymentError, MlemObjectNotFound
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    DeployState,
    DeployStatus,
    MlemDeployment,
    MlemModel,
)
from mlem.telemetry import pass_telemetry_params
from mlem.ui import EMOJI_STOP, echo, no_echo, set_echo

logger = logging.getLogger(__name__)

deployment = Typer(
    name="deployment",
    help="A set of commands to set up and manage deployments",
    cls=mlem_group("runtime", aliases=["deploy"]),
)
app.add_typer(deployment)

deploy_run = Typer(
    name="run",
    help="""Deploy a model to a target environment. Can use an existing deployment
    declaration or create a new one on-the-fly.
    """,
    cls=mlem_group("other"),
    subcommand_metavar="deployment",
)
deployment.add_typer(deploy_run)


@mlem_group_callback(deploy_run, required=["model", "load"])
def deploy_run_callback(
    load: str = option_load("deployment"),
    model: str = make_not_required(option_model),
    model_project: Optional[str] = option_model_project,
    model_rev: Optional[str] = option_model_rev,
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
):
    """Deploy a model to a target environment. Can use an existing deployment
    declaration or create a new one on-the-fly.
    """
    from mlem.api.commands import deploy

    mlem_model = load_meta(
        model, project=model_project, rev=model_rev, force_type=MlemModel
    )
    with pass_telemetry_params():
        deploy(
            load,
            mlem_model,
            project=project,
            rev=rev,
        )


@for_each_impl(MlemDeployment)
def create_deploy_run_command(type_name):
    @mlem_command(
        type_name,
        section="deployments",
        parent=deploy_run,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=add_env_params_deployment(
            type_name, MlemDeployment
        ),
        hidden=type_name.startswith("_"),
        lazy_help=lazy_class_docstring(MlemDeployment.object_type, type_name),
        no_pass_from_parent=["file_conf"],
        is_generated_from_ext=True,
    )
    def deploy_run_command(
        path: str = Argument(
            ..., help="Where to save the object (.mlem file)"
        ),
        model: str = make_not_required(option_model),
        model_project: Optional[str] = option_model_project,
        model_rev: Optional[str] = option_model_rev,
        project: Optional[str] = option_project,
        file_conf: List[str] = option_file_conf("deployment"),
        force: bool = Option(
            False, is_flag=True, help="Force re-create if parameters changed"
        ),
        **__kwargs__,
    ):
        from mlem.api.commands import deploy

        if model is None:
            raise click.MissingParameter(
                param=get_click_param(
                    ParamMeta(
                        name="model", default=option_model, annotation=str
                    )
                )[0]
            )
        __kwargs__ = process_fields(type_name, MlemDeployment, __kwargs__)
        _meta = build_mlem_object(
            MlemDeployment,
            type_name,
            str_conf=None,
            file_conf=file_conf,
            **__kwargs__,
        )
        try:
            meta = load_meta(path, project=project, force_type=MlemDeployment)
            if meta != _meta:
                if not meta.is_state_empty:
                    if not force:
                        raise DeploymentError(
                            f"Different deployment meta already exists at {meta.loc}. Please use `mlem deployment run --load <path> ...`"
                        )
                    echo(
                        EMOJI_STOP
                        + "Removing old deployment since parameters changed"
                    )
                    meta.remove()
                with wrap_build_error(type_name, MlemDeployment):
                    meta = _meta.dump(path, project=project)
        except MlemObjectNotFound:
            with wrap_build_error(type_name, MlemDeployment):
                meta = _meta.dump(path, project=project)
        mlem_model = load_meta(
            model,
            project=model_project,
            rev=model_rev,
            force_type=MlemModel,
        )
        with pass_telemetry_params():
            deploy(
                meta,
                mlem_model,
                project=project,
            )


@mlem_command("remove", parent=deployment)
def deploy_remove(
    path: str = Argument(..., help="Path to deployment meta"),
    project: Optional[str] = option_project,
):
    """Stop and destroy deployed instance."""
    deploy_meta = load_meta(path, project=project, force_type=MlemDeployment)
    deploy_meta.remove()


@mlem_command("status", parent=deployment)
def deploy_status(
    path: str = Argument(..., help="Path to deployment meta"),
    project: Optional[str] = option_project,
):
    """Print status of deployed service."""
    with no_echo():
        deploy_meta = load_meta(
            path, project=project, force_type=MlemDeployment
        )
        status = deploy_meta.get_status()
    echo(status)


@mlem_command("wait", parent=deployment)
def deploy_wait(
    path: str = Argument(..., help="Path to deployment meta"),
    project: Optional[str] = option_project,
    statuses: List[DeployStatus] = Option(
        [DeployStatus.RUNNING],
        "-s",
        "--status",
        help="statuses to wait for",
    ),
    intermediate: List[DeployStatus] = Option(
        None, "-i", "--intermediate", help="Possible intermediate statuses"
    ),
    poll_timeout: float = Option(
        1.0, "-p", "--poll-timeout", help="Timeout between attempts"
    ),
    times: int = Option(
        0, "-t", "--times", help="Number of attempts. 0 -> indefinite"
    ),
):
    """Wait for status of deployed service"""
    with no_echo():
        deploy_meta = load_meta(
            path, project=project, force_type=MlemDeployment
        )
        deploy_meta.wait_for_status(
            statuses, poll_timeout, times, allowed_intermediate=intermediate
        )


@mlem_command("apply", parent=deployment)
def deploy_apply(
    path: str = Argument(..., help="Path to deployment meta"),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    data: str = Argument(..., help="Path to data object"),
    data_project: Optional[str] = option_data_project,
    data_rev: Optional[str] = option_data_rev,
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    target_project: Optional[str] = option_target_project,
    method: str = option_method,
    json: bool = option_json,
):
    """Apply a deployed model to data."""
    with set_echo(None if json else ...):
        deploy_meta = load_meta(
            path, project=project, rev=rev, force_type=MlemDeployment
        )
        state: DeployState = deploy_meta.get_state()
        if (
            state == deploy_meta.state_type(declaration=deploy_meta)
            and not deploy_meta.state_type.allow_default
        ):
            raise DeploymentError(
                f"{deploy_meta.type} deployment has no state. Either {deploy_meta.type} is not deployed yet or has been un-deployed again."
            )
        client = deploy_meta.get_client(state)

        result = run_apply_remote(
            client,
            data,
            data_project,
            data_rev,
            method,
            output,
            target_project,
        )
    if output is None:
        print(
            dumps(
                DataAnalyzer.analyze(result).get_serializer().serialize(result)
            )
        )
