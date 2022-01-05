import os
import posixpath

import click
from fsspec.implementations.local import LocalFileSystem

from mlem.cli.main import mlem_command
from mlem.core.metadata import load_meta
from mlem.core.objects import (
    DeployMeta,
    ModelMeta,
    TargetEnvMeta,
    mlem_dir_path,
)

DEPLOY_EXT = ".deployed.yaml"


@mlem_command(
    context_settings={
        "ignore_unknown_options": True,
    }
)
@click.argument("model")
@click.argument("target_environment")
@click.argument("deploy_args", nargs=-1, type=click.UNPROCESSED)
def deploy(model, target_environment, deploy_args):
    model_meta = load_meta(model, force_type=ModelMeta)
    env_meta = load_meta(target_environment, force_type=TargetEnvMeta)
    if len(env_meta.additional_args) != len(deploy_args):
        raise ValueError(
            f"Invalid arguments for {env_meta.alias} deploy: {env_meta.additional_args} needed"
        )
    args = dict(zip(env_meta.additional_args, deploy_args))

    previous = DeployMeta.find(target_environment, model, False)
    if previous is not None:
        if not isinstance(previous.deployment, env_meta.deployment_type):
            raise ValueError(
                f"Cant redeploy {previous.deployment.__class__} to {env_meta.__class__}"
            )
        click.echo("Already deployed, updating")
        deployment = env_meta.update(model_meta, previous.deployment)
    else:
        deployment = env_meta.deploy(model_meta, **args)

    deploy_meta = DeployMeta(
        env_path=target_environment, model_path=model, deployment=deployment
    )
    deploy_meta.dump(posixpath.join(target_environment, model))


@mlem_command()
@click.argument("target_environment")
@click.argument("model")
def destroy(target_environment, model):
    deployed = posixpath.join(target_environment, model)
    deploy_meta = load_meta(deployed, force_type=DeployMeta)
    deploy_meta.deployment.destroy()
    os.unlink(mlem_dir_path(deployed, LocalFileSystem(), obj_type=DeployMeta))


@mlem_command()
@click.argument("target_environment")
@click.argument("model")
def status(target_environment, model):
    deploy_meta = DeployMeta.find(target_environment, model)
    print(deploy_meta.deployment.get_status())
