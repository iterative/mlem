import click

from mlem.cli.main import config_arg, mlem_command, with_model_meta
from mlem.core.objects import ModelMeta
from mlem.pack import Packager


@mlem_command()
@with_model_meta
@config_arg("packager", model=Packager)
@click.argument("out")
def pack(model: ModelMeta, packager: Packager, out: str):
    """\b
    Pack model.
    Packager: either "docker_dir" or "docker".
    Out: path in case of "docker_dir" and image name in case of "docker".
    """
    from mlem.api.commands import pack

    pack(packager, model, out)
