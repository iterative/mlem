from typing import List, Optional

from typer import Argument, Typer

from mlem.cli.main import (
    app,
    mlem_command,
    mlem_group,
    option_conf,
    option_file_conf,
    option_load,
    option_project,
    option_rev,
)
from mlem.cli.utils import (
    abc_fields_parameters,
    config_arg,
    for_each_impl,
    lazy_class_docstring,
)
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemBuilder, MlemModel

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


@for_each_impl(MlemBuilder)
def create_build_command(type_name):
    @mlem_command(
        type_name,
        section="builders",
        parent=build,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(
            type_name, MlemBuilder
        ),
        hidden=type_name.startswith("_"),
        lazy_help=lazy_class_docstring(MlemBuilder.abs_name, type_name),
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
