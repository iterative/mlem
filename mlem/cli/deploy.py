from json import dumps
from typing import List, Optional

from typer import Argument, Option, Typer

from mlem.cli.apply import run_apply_remote
from mlem.cli.main import (
    MlemGroupSection,
    app,
    mlem_command,
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
from mlem.core.objects import MlemDeploy
from mlem.ui import echo, no_echo, set_echo

deploy = Typer(
    name="deploy", help="Manage deployments", cls=MlemGroupSection("runtime")
)
app.add_typer(deploy)


@mlem_command("create", parent=deploy)
def deploy_create(
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
    conf: Optional[List[str]] = Option(
        None,
        "-c",
        "--conf",
        help="Configuration for new deployment meta if it does not exist",
    ),
):
    """Deploy a model to target environment. Can use existing deployment declaration or create a new one on-the-fly

    Examples:
        Create new deployment
        $ mlem create env heroku staging -c api_key=...
        $ mlem deploy create service_name -m model -t staging -c name=my_service

        Deploy existing meta
        $ mlem create env heroku staging -c api_key=...
        $ mlem create deployment heroku service_name -c app_name=my_service -c model=model -c env=staging
        $ mlem deploy create service_name
    """
    from mlem.api.commands import deploy

    deploy(
        path,
        model,
        env,
        project,
        external=external,
        index=index,
        **parse_string_conf(conf or []),
    )


@mlem_command("teardown", parent=deploy)
def deploy_teardown(
    path: str = Argument(..., help="Path to deployment meta"),
    project: Optional[str] = option_project,
):
    """Stop and destroy deployed instance

    Examples:
        $ mlem deploy teardown service_name
    """
    deploy_meta = load_meta(path, project=project, force_type=MlemDeploy)
    deploy_meta.destroy()


@mlem_command("status", parent=deploy)
def deploy_status(
    path: str = Argument(..., help="Path to deployment meta"),
    project: Optional[str] = option_project,
):
    """Print status of deployed service

    Examples:
        $ mlem deploy status service_name
    """
    with no_echo():
        deploy_meta = load_meta(path, project=project, force_type=MlemDeploy)
        status = deploy_meta.get_status()
    echo(status)


@mlem_command("apply", parent=deploy)
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
    """Apply method of deployed service

    Examples:
        $ mlem deploy apply service_name
    """

    with set_echo(None if json else ...):
        deploy_meta = load_meta(
            path, project=project, rev=rev, force_type=MlemDeploy
        )
        if deploy_meta.state is None:
            raise DeploymentError(
                f"{deploy_meta.type} deployment has no state. Either {deploy_meta.type} is not deployed yet or has been un-deployed again."
            )
        client = deploy_meta.state.get_client()

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
