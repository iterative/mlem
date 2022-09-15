from json import dumps
from typing import List, Optional

from typer import Argument, Option, Typer

from mlem.cli.apply import run_apply_remote
from mlem.cli.main import (
    app,
    mlem_command,
    mlem_group,
    option_conf,
    option_data_project,
    option_data_rev,
    option_external,
    option_index,
    option_json,
    option_method,
    option_project,
    option_rev,
    option_target_project,
)
from mlem.core.base import parse_string_conf
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import DeploymentError
from mlem.core.metadata import load_meta
from mlem.core.objects import DeployState, DeployStatus, MlemDeployment
from mlem.ui import echo, no_echo, set_echo

deployment = Typer(
    name="deployment",
    help="A set of commands to set up and manage deployments.",
    cls=mlem_group("runtime", aliases=["deploy"]),
)
app.add_typer(deployment)


@mlem_command("run", parent=deployment)
def deploy_run(
    path: str = Argument(
        ...,
        help="Path to deployment meta (will be created if it does not exist)",
    ),
    model: Optional[str] = Option(None, "-m", "--model", help="Path to model"),
    env: Optional[str] = Option(
        None, "-t", "--env", help="Path to target environment"
    ),
    project: Optional[str] = option_project,
    external: bool = option_external,
    index: bool = option_index,
    conf: Optional[List[str]] = option_conf(),
):
    """Deploy a model to a target environment. Can use an existing deployment
    declaration or create a new one on-the-fly.
    """
    from mlem.api.commands import deploy

    conf = conf or []
    env_conf = [c[len("env.") :] for c in conf if c.startswith("env.")]
    conf = [c for c in conf if not c.startswith("env.")]
    deploy(
        path,
        model,
        env,
        project,
        external=external,
        index=index,
        env_kwargs=parse_string_conf(env_conf),
        **parse_string_conf(conf or []),
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
    index: bool = option_index,
    json: bool = option_json,
):
    """Apply a deployed model to data."""
    with set_echo(None if json else ...):
        deploy_meta = load_meta(
            path, project=project, rev=rev, force_type=MlemDeployment
        )
        state: DeployState = deploy_meta.get_state()
        if (
            state == deploy_meta.state_type()
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
            index,
            method,
            output,
            target_project,
        )
    if output is None and json:
        print(
            dumps(
                DataAnalyzer.analyze(result).get_serializer().serialize(result)
            )
        )
