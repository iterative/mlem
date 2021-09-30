import os

import click

from mlem.cli.main import cli
from mlem.core.objects import (
    DeployMeta,
    ModelMeta,
    TargetEnvMeta,
    mlem_dir_path,
)

DEPLOY_EXT = ".deployed.yaml"


@cli.command(
    context_settings={
        "ignore_unknown_options": True,
    }
)
@click.argument("model")
@click.argument("target_environment")
@click.argument("deploy_args", nargs=-1, type=click.UNPROCESSED)
def deploy(model, target_environment, deploy_args):
    model_meta = ModelMeta.read(model)
    env_meta = TargetEnvMeta.read(target_environment)
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

    deploy_meta = DeployMeta(target_environment, model, deployment)
    deploy_meta.dump_meta(os.path.join(target_environment, model))


@cli.command()
@click.argument("target_environment")
@click.argument("model")
def destroy(target_environment, model):
    deployed = os.path.join(target_environment, model)
    deploy_meta = DeployMeta.read(deployed)
    deploy_meta.deployment.destroy()
    os.unlink(mlem_dir_path(deployed, obj_type=DeployMeta))


@cli.command()
@click.argument("target_environment")
@click.argument("model")
def status(target_environment, model):
    deploy_meta = DeployMeta.find(target_environment, model)
    print(deploy_meta.deployment.get_status())
