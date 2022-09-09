from typing import List, Optional

from typer import Option, Typer

from mlem.cli.main import (
    app,
    mlem_command,
    mlem_group,
    mlem_group_callback,
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
from mlem.core.objects import MlemModel
from mlem.runtime.server import Server

serve = Typer(
    name="serve",
    help="""Create an API from model methods using a server implementation.

    Examples:
        $ mlem serve fastapi https://github.com/iterative/example-mlem/models/logreg
    """,
    cls=mlem_group("runtime"),
    subcommand_metavar="server",
)
app.add_typer(serve)


@mlem_group_callback(serve, required=["model", "load"])
def serve_load(
    model: str = Option(
        None, "-m", "--model", help="Path to MLEM model object"
    ),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    load: Optional[str] = option_load("server"),
):
    from mlem.api.commands import serve

    serve(
        load_meta(model, project, rev, force_type=MlemModel),
        config_arg(
            Server,
            load,
            None,
            conf=None,
            file_conf=None,
        ),
    )


@for_each_impl(Server)
def create_serve_command(type_name):
    @mlem_command(
        type_name,
        section="servers",
        parent=serve,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(type_name, Server),
        hidden=type_name.startswith("_"),
        lazy_help=lazy_class_docstring(Server.abs_name, type_name),
        no_pass_from_parent=["file_conf"],
    )
    def serve_command(
        model: str = Option(
            ..., "-m", "--model", help="Model to create service from"
        ),
        project: Optional[str] = option_project,
        rev: Optional[str] = option_rev,
        file_conf: List[str] = option_file_conf("server"),
        **__kwargs__
    ):
        from mlem.api.commands import serve

        serve(
            load_meta(model, project, rev, force_type=MlemModel),
            config_arg(
                Server,
                None,
                type_name,
                conf=None,
                file_conf=file_conf,
                **__kwargs__
            ),
        )
