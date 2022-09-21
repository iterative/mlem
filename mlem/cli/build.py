from typing import List, Optional

from typer import Typer

from mlem.cli.main import (
    app,
    mlem_command,
    mlem_group,
    mlem_group_callback,
    option_file_conf,
    option_load,
    option_model,
    option_project,
    option_rev,
)
from mlem.cli.utils import (
    abc_fields_parameters,
    config_arg,
    for_each_impl,
    lazy_class_docstring,
    make_not_required,
)
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemBuilder, MlemModel

build = Typer(
    name="build",
    help="""
        Build models into re-usable assets you can distribute and use in production,
such as a Docker image or Python package.
        """,
    cls=mlem_group("runtime", aliases=["export"]),
    subcommand_metavar="builder",
)
app.add_typer(build)


@mlem_group_callback(build, required=["model", "load"])
def build_load(
    model: str = make_not_required(option_model),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    load: str = option_load("builder"),
):
    from mlem.api.commands import build

    build(
        config_arg(
            MlemBuilder,
            load,
            None,
            conf=None,
            file_conf=None,
        ),
        load_meta(model, project, rev, force_type=MlemModel),
    )


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
        no_pass_from_parent=["file_conf"],
    )
    def build_type(
        model: str = option_model,
        project: Optional[str] = option_project,
        rev: Optional[str] = option_rev,
        file_conf: List[str] = option_file_conf("builder"),
        **__kwargs__
    ):
        from mlem.api.commands import build

        build(
            config_arg(
                MlemBuilder,
                None,
                type_name,
                conf=None,
                file_conf=file_conf,
                **__kwargs__
            ),
            load_meta(model, project, rev, force_type=MlemModel),
        )
