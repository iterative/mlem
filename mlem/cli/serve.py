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
from mlem.cli.utils import abc_fields_parameters, config_arg
from mlem.core.base import load_impl_ext
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemModel
from mlem.runtime.server import Server
from mlem.utils.entrypoints import list_implementations

serve = Typer(
    name="serve",
    help="""Serve selected model

    Examples:
        $ mlem serve fastapi https://github.com/iterative/example-mlem/models/logreg
    """,
    cls=mlem_group("runtime"),
    subcommand_metavar="server",
)
app.add_typer(serve)


def create_serve(type_name, cls):
    @mlem_command(
        type_name,
        section="servers",
        parent=serve,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(cls),
        hidden=type_name.startswith("_"),
        help=cls.__doc__,
    )
    def serve_command(
        model: str = Argument(..., help="Model to create service from"),
        project: Optional[str] = option_project,
        rev: Optional[str] = option_rev,
        load: Optional[str] = option_load("server"),
        conf: List[str] = option_conf("server"),
        file_conf: List[str] = option_file_conf("server"),
        **__kwargs__
    ):
        from mlem.api.commands import serve

        serve(
            load_meta(model, project, rev, force_type=MlemModel),
            config_arg(Server, load, type_name, conf, file_conf, **__kwargs__),
        )

    serve_command.__doc__ = cls.__doc__


any_implementations = False
for server_type_name in list_implementations(Server):
    try:
        server_class = load_impl_ext(Server.abs_name, server_type_name)
        create_serve(server_type_name, server_class)
        any_implementations = True
    except ImportError:
        pass

if not any_implementations:
    serve.info.help += """\nNo available server implementations :("""
