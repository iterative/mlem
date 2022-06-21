from json import dumps
from typing import List, Optional

from typer import Argument, Option

from mlem.api import import_object
from mlem.cli.main import (
    config_arg,
    mlem_command,
    option_conf,
    option_data_project,
    option_data_rev,
    option_external,
    option_file_conf,
    option_index,
    option_json,
    option_load,
    option_method,
    option_project,
    option_rev,
    option_target_project,
)
from mlem.core.data_type import DataAnalyzer
from mlem.core.errors import UnsupportedDataBatchLoading
from mlem.core.import_objects import ImportHook
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemData, MlemModel
from mlem.runtime.client import Client
from mlem.ui import set_echo
from mlem.utils.entrypoints import list_implementations


@mlem_command("apply", section="runtime")
def apply(
    model: str = Argument(..., help="Path to model object"),
    data_path: str = Argument(..., metavar="data", help="Path to data object"),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    method: str = option_method,
    data_project: Optional[str] = option_data_project,
    data_rev: Optional[str] = option_data_rev,
    import_: bool = Option(
        False,
        "-i",
        "--import",
        help="Try to import data on-the-fly",
    ),
    import_type: str = Option(
        None,
        "--import-type",
        "--it",
        # TODO: change ImportHook to MlemObject to support ext machinery
        help=f"Specify how to read data file for import. Available types: {list_implementations(ImportHook)}",
    ),
    batch_size: Optional[int] = Option(
        None,
        "-b",
        "--batch_size",
        help="Batch size for reading data in batches.",
    ),
    index: bool = option_index,
    external: bool = option_external,
    json: bool = option_json,
):
    """Apply a model to data. Resulting data will be saved as MLEM object to `output` if it is provided, otherwise will be printed

    Examples:
        Apply local mlem model to local mlem data
        $ mlem apply mymodel mydata --method predict --output myprediction

        Apply local mlem model to local data file
        $ mlem apply mymodel data.csv --method predict --import --import-type pandas[csv] --output myprediction

        Apply a version of remote model to a version of remote data
        $ mlem apply models/logreg --project https://github.com/iterative/example-mlem --rev main
                     data/test_x --data-project https://github.com/iterative/example-mlem --data-rev main
                     --method predict --output myprediction
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
            index=index,
            external=external,
            batch_size=batch_size,
        )
    if output is None and json:
        print(
            dumps(
                DataAnalyzer.analyze(result).get_serializer().serialize(result)
            )
        )


@mlem_command("apply-remote", section="runtime")
def apply_remote(
    subtype: str = Argument(
        "",
        help=f"Type of client. Choices: {list_implementations(Client)}",
        show_default=False,
    ),
    data: str = Argument(..., help="Path to data object"),
    project: Optional[str] = option_project,
    rev: Optional[str] = option_rev,
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    target_project: Optional[str] = option_target_project,
    method: str = option_method,
    index: bool = option_index,
    json: bool = option_json,
    load: Optional[str] = option_load("client"),
    conf: List[str] = option_conf("client"),
    file_conf: List[str] = option_file_conf("client"),
):
    """Apply a model (deployed somewhere remotely) to data. Resulting data will be saved as MLEM object to `output` if it is provided, otherwise will be printed

    Examples:
        Apply hosted mlem model to local mlem data
        $ mlem apply-remote http mydata -c host="0.0.0.0" -c port=8080 --output myprediction
    """
    client = config_arg(Client, load, subtype, conf, file_conf)

    with set_echo(None if json else ...):
        result = run_apply_remote(
            client, data, project, rev, index, method, output, target_project
        )
    if output is None and json:
        print(
            dumps(
                DataAnalyzer.analyze(result).get_serializer().serialize(result)
            )
        )


def run_apply_remote(
    client: Client,
    data_path: str,
    project,
    rev,
    index,
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
        index=index,
    )
    return result
