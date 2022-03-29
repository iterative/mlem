from typing import List, Optional

import click
from typer import Argument, Option

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_link,
    option_repo,
)
from mlem.core.base import parse_string_conf
from mlem.core.metadata import load_meta
from mlem.core.objects import DeployMeta


@mlem_command()
def deploy(
    path: str = Argument(...),
    model: Optional[str] = Option(None, "-m", "--model"),
    env: Optional[str] = Option(None, "-t", "--env"),
    repo: Optional[str] = option_repo,
    external: bool = option_external,
    link: bool = option_link,
    conf: Optional[List[str]] = Option(None, "-c", "--conf"),
):
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


@mlem_command()
def destroy(path: str = Argument(...), repo: Optional[str] = option_repo):
    deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
    deploy_meta.destroy()


@mlem_command()
def status(path: str = Argument(...), repo: Optional[str] = option_repo):
    deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
    click.echo(deploy_meta.get_status())
