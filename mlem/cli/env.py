import click

from ..core.objects import TargetEnvMeta
from .main import cli
from .utils import create_configurable


@cli.group("env")
def environment():
    pass


@environment.command()
@click.argument("name")
@click.argument("type")
def create(name, type):
    ENV_CLASSES = {
        cls.alias: type
        for type, cls in TargetEnvMeta._subtypes.items()
        if cls.alias is not ...
    }
    env_meta = create_configurable(TargetEnvMeta, type, ENV_CLASSES)
    env_meta.dump(name)
