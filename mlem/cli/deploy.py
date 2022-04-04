from typing import List, Optional

from typer import Argument, Option, Typer

from mlem.cli.main import (
    MlemGroupSection,
    app,
    mlem_command,
    option_external,
    option_link,
    option_repo,
)
from mlem.core.base import parse_string_conf
from mlem.core.metadata import load_meta
from mlem.core.objects import DeployMeta
from mlem.ui import echo, no_echo

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
        **parse_string_conf(conf or [])
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
