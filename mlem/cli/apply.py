from json import dumps
from typing import List, Optional

from typer import Argument, Option, Typer

from mlem.api import import_object
from mlem.cli.main import (
    PATH_METAVAR,
    app,
    mlem_command,
    mlem_group,
    mlem_group_callback,
    option_data,
    option_data_project,
    option_data_rev,
    option_file_conf,
    option_json,
    option_load,
    option_method,
    option_project,
    option_rev,
    option_target_project,
)
from mlem.cli.utils import (
    abc_fields_parameters,
    config_arg,
    for_each_impl,
    lazy_class_docstring,
    make_not_required,
)
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import UnsupportedDataBatchLoading
from mlem.core.import_objects import ImportHook
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemData, MlemModel
from mlem.runtime.client import Client
from mlem.ui import set_echo
from mlem.utils.entrypoints import list_implementations

option_output = Option(
    None,
    "-o",
    "--output",
    help="Where to save model outputs",
    metavar=PATH_METAVAR,
)
option_import = Option(
    False,
    "-i",
    "--import",
    help="Try to import data on-the-fly",
)
option_import_type = Option(
    None,
    "--import-type",
    "--it",
    # TODO: change ImportHook to MlemObject to support ext machinery
    help=f"Specify how to read data file for import. Available types: {list_implementations(ImportHook)}",
)
option_batch_size = Option(
    None,
    "-b",
    "--batch_size",
    help="Batch size for reading data in batches",
)


@mlem_command("apply", section="runtime")
def apply(
    model: str = Argument(..., metavar="model", help="Path to model object"),
    data_path: str = Argument(..., metavar="data", help="Path to data object"),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    output: Optional[str] = option_output,
    method: str = option_method,
    data_project: Optional[str] = option_data_project,
    data_rev: Optional[str] = option_data_rev,
    import_: bool = option_import,
    import_type: str = option_import_type,
    batch_size: Optional[int] = option_batch_size,
    json: bool = option_json,
):
    """Apply a model to data. The result will be saved as a MLEM object to `output` if
    provided. Otherwise, it will be printed to `stdout`.
    """
    from mlem.api import apply

    with set_echo(None if json else ...):
        if import_:
            if batch_size:
                raise UnsupportedDataBatchLoading(
                    "Batch data loading is currently not supported for loading data on-the-fly"
                )
            data = import_object(
                data_path,
                project=data_project,
                rev=data_rev,
                type_=import_type,
            )
        else:
            data = load_meta(
                data_path,
                data_project,
                data_rev,
                load_value=batch_size is None,
                force_type=MlemData,
            )
        meta = load_meta(model, project, rev, force_type=MlemModel)

        result = apply(
            meta,
            data,
            method=method,
            output=output,
            batch_size=batch_size,
        )
    if output is None:
        print(
            dumps(
                DataAnalyzer.analyze(result).get_serializer().serialize(result)
            )
        )


apply_remote = Typer(
    name="apply-remote",
    help="""Apply a deployed-model (possibly remotely) to data. The results will be saved as
a MLEM object to `output` if provided. Otherwise, it will be printed to
`stdout`.
    """,
    cls=mlem_group("runtime"),
    subcommand_metavar="client",
)
app.add_typer(apply_remote)


def _apply_remote(
    data,
    project,
    rev,
    method,
    output,
    target_project,
    json,
    type_name,
    load,
    file_conf,
    kwargs,
):
    client = config_arg(
        Client,
        load,
        type_name,
        conf=None,
        file_conf=file_conf,
        **(kwargs or {}),
    )

    with set_echo(None if json else ...):
        result = run_apply_remote(
            client,
            data,
            project,
            rev,
            method,
            output,
            target_project,
        )
    if output is None:
        print(
            dumps(
                DataAnalyzer.analyze(result).get_serializer().serialize(result)
            )
        )


@mlem_group_callback(apply_remote, required=["data", "load"])
def apply_remote_load(
    data: str = make_not_required(option_data),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    output: Optional[str] = option_output,
    target_project: Optional[str] = option_target_project,
    method: str = option_method,
    json: bool = option_json,
    load: Optional[str] = option_load("client"),
):
    return _apply_remote(
        data,
        project,
        rev,
        method,
        output,
        target_project,
        json,
        None,
        load,
        None,
        None,
    )


@for_each_impl(Client)
def create_apply_remote(type_name):
    @mlem_command(
        type_name,
        section="clients",
        parent=apply_remote,
        dynamic_metavar="__kwargs__",
        dynamic_options_generator=abc_fields_parameters(type_name, Client),
        hidden=type_name.startswith("_"),
        lazy_help=lazy_class_docstring(Client.abs_name, type_name),
        no_pass_from_parent=["file_conf"],
    )
    def apply_remote_func(
        data: str = option_data,
        project: Optional[str] = option_project,
        rev: Optional[str] = option_rev,
        output: Optional[str] = option_output,
        target_project: Optional[str] = option_target_project,
        method: str = option_method,
        json: bool = option_json,
        file_conf: List[str] = option_file_conf("client"),
        **__kwargs__,
    ):
        return _apply_remote(
            data,
            project,
            rev,
            method,
            output,
            target_project,
            json,
            type_name,
            None,
            file_conf,
            __kwargs__,
        )


def run_apply_remote(
    client: Client,
    data_path: str,
    project,
    rev,
    method,
    output,
    target_project,
):
    from mlem.api import apply_remote

    data = load_meta(
        data_path,
        project,
        rev,
        load_value=True,
        force_type=MlemData,
    )
    result = apply_remote(
        client,
        data,
        method=method,
        output=output,
        target_project=target_project,
    )
    return result
