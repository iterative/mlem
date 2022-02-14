from typing import List

import click

from mlem.cli.main import (
    mlem_command,
    option_external,
    option_link,
    option_repo,
)
from mlem.core.base import parse_string_conf
from mlem.core.metadata import load_meta
from mlem.core.objects import DeployMeta

DEPLOY_EXT = ".deployed.yaml"


@mlem_command()
@click.argument("path")
@click.option("-m", "--model")
@click.option("-t", "--env")
@click.option("-c", "--conf", multiple=True)
@option_repo
@option_external
@option_link
def deploy(path, model, env, repo, external, link, conf: List[str]):
    from mlem.api.commands import deploy

    deploy(
        path,
        model,
        env,
        repo,
        external=external,
        link=link,
        **parse_string_conf(conf)
    )


@mlem_command()
@click.argument("path")
@option_repo
def destroy(path, repo):
    deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
    deploy_meta.destroy()


@mlem_command()
@click.argument("path")
@option_repo
def status(path, repo):
    deploy_meta = load_meta(path, repo=repo, force_type=DeployMeta)
    click.echo(deploy_meta.get_status())
