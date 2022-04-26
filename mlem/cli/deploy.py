from json import dumps
from typing import List, Optional

from typer import Argument, Option, Typer

from mlem.cli.main import (
    MlemGroupSection,
    app,
    mlem_command,
    option_external,
    option_json,
    option_link,
    option_method,
    option_repo,
)
from mlem.core.base import parse_string_conf
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.errors import DeploymentError
from mlem.core.metadata import load_meta
from mlem.core.objects import DatasetMeta, DeployMeta
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
    repo: Optional[str] = option_repo,
    external: bool = option_external,
    link: bool = option_link,
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
        repo,
        external=external,
        link=link,
        **parse_string_conf(conf or []),
    )


@mlem_command("teardown", parent=deploy)
def deploy_teardown(
    path: str = Argument(..., help="Path to deployment meta"),
    repo: Optional[str] = option_repo,
):
    """Stop and destroy deployed instance

    Examples:
        $ mlem deploy teardown service_name
    """
    deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
    deploy_meta.destroy()


@mlem_command("status", parent=deploy)
def deploy_status(
    path: str = Argument(..., help="Path to deployment meta"),
    repo: Optional[str] = option_repo,
):
    """Print status of deployed service

    Examples:
        $ mlem deploy status service_name
    """
    with no_echo():
        deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
        status = deploy_meta.get_status()
    echo(status)


@mlem_command("apply", parent=deploy)
def deploy_apply(
    path: str = Argument(..., help="Path to deployment meta"),
    data: str = Argument(..., help="Path to dataset object"),
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    method: str = option_method,
    link: bool = option_link,
    json: bool = option_json,
    repo: Optional[str] = option_repo,
):
    """Apply method of deployed service

    Examples:
        $ mlem deploy apply service_name
    """
    from mlem.api import apply_remote

    with set_echo(None if json else ...):
        deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
        if deploy_meta.state is None:
            raise DeploymentError(
                f"{deploy_meta.type} deployment has no state. Either {deploy_meta.type} is not deployed yet or has been un-deployed again."
            )
        client = deploy_meta.state.get_client()

        dataset = load_meta(
            data,
            None,
            None,
            load_value=True,
            force_type=DatasetMeta,
        )
        result = apply_remote(
            client,
            dataset,
            method=method,
            output=output,
            link=link,
        )
    if output is None and json:
        print(
            dumps(
                DatasetAnalyzer.analyze(result)
                .get_serializer()
                .serialize(result)
            )
        )
