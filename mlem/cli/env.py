import click

from ..core.objects import TargetEnvMeta
from .main import cli, verbose_option
from .utils import create_configurable


@cli.group("env")
def environment():
    pass


@environment.command()
@verbose_option
@click.argument("name")
@click.argument("type")
def create(name, type):
    ENV_CLASSES = {
        cls.alias: type
        for type, cls in TargetEnvMeta.__type_map__.items()
        if cls.alias is not ...
    }
    env_meta = create_configurable(TargetEnvMeta, type, ENV_CLASSES)
    env_meta.dump(name)
