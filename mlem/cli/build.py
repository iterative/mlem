from typing import List, Optional

from typer import Argument, Typer

from mlem.cli.main import (
    abc_fields_parameters,
    app,
    config_arg,
    mlem_command,
    mlem_group,
    option_conf,
    option_file_conf,
    option_load,
    option_project,
    option_rev,
)
from mlem.core.base import load_impl_ext
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.utils.entrypoints import list_implementations

build = Typer(
    name="build",
    help="""
        Build/export model

        Examples:
            Build docker image from model
            $ mlem build mymodel docker -c server.type=fastapi -c image.name=myimage

            Create build docker_dir declaration and build it
            $ mlem declare builder docker_dir -c server=fastapi -c target=build build_dock
            $ mlem build mymodel --load build_dock
        """,
    cls=mlem_group("runtime", aliases=["export"]),
    subcommand_metavar="builder",
)
app.add_typer(build)


def create_build_command(type_name, cls):
    @mlem_command(
        type_name,
        section="builders",
        parent=build,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(cls),
    )
    def build_type(
        model: str = Argument(..., help="Path to model"),
        project: Optional[str] = option_project,
        rev: Optional[str] = option_rev,
        load: Optional[str] = option_load("builder"),
        conf: List[str] = option_conf("builder"),
        file_conf: List[str] = option_file_conf("builder"),
        **__kwargs__
    ):
        from mlem.api.commands import build

        build(
            config_arg(
                MlemBuilder, load, type_name, conf, file_conf, **__kwargs__
            ),
            load_meta(model, project, rev, force_type=MlemModel),
        )

    build_type.__doc__ = cls.__doc__


any_implementations = False
for builder_type_name in list_implementations(MlemBuilder):
    try:
        builder_class = load_impl_ext(MlemBuilder.abs_name, builder_type_name)
        create_build_command(builder_type_name, builder_class)
        any_implementations = True
    except ImportError:
        pass

if not any_implementations:
    build.info.help += """\nNo available builder implementations :("""
