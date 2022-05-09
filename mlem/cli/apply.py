from json import dumps
from typing import List, Optional

from typer import Argument, Option

from mlem.api import import_object
from mlem.cli.main import (
    config_arg,
    mlem_command,
    option_conf,
    option_data_repo,
    option_data_rev,
    option_external,
    option_file_conf,
    option_index,
    option_json,
    option_load,
    option_method,
    option_repo,
    option_rev,
    option_target_repo,
)
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.errors import UnsupportedDatasetBatchLoading
from mlem.core.import_objects import ImportHook
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemDataset, MlemModel
from mlem.ext import list_implementations
from mlem.runtime.client.base import BaseClient
from mlem.ui import set_echo


@mlem_command("apply", section="runtime")
def apply(
    model: str = Argument(..., help="Path to model object"),
    data: str = Argument(..., help="Path to dataset object"),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    method: str = option_method,
    data_repo: Optional[str] = option_data_repo,
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
    """Apply a model to a dataset. Resulting dataset will be saved as MLEM object to `output` if it is provided, otherwise will be printed

    Examples:
        Apply local mlem model to local mlem dataset
        $ mlem apply mymodel mydataset --method predict --output myprediction

        Apply local mlem model to local data file
        $ mlem apply mymodel data.csv --method predict --import --import-type pandas[csv] --output myprediction

        Apply a version of remote model to a version of remote dataset
        $ mlem apply models/logreg --repo https://github.com/iterative/example-mlem --rev main
                     data/test_x --data-repo https://github.com/iterative/example-mlem --data-rev main
                     --method predict --output myprediction
    """
    from mlem.api import apply

    with set_echo(None if json else ...):
        if import_:
            if batch_size:
                raise UnsupportedDatasetBatchLoading(
                    "Batch data loading is currently not supported for loading data on-the-fly"
                )
            dataset = import_object(
                data, repo=data_repo, rev=data_rev, type_=import_type
            )
        else:
            dataset = load_meta(
                data,
                data_repo,
                data_rev,
                load_value=batch_size is None,
                force_type=MlemDataset,
            )
        meta = load_meta(model, repo, rev, force_type=MlemModel)

        result = apply(
            meta,
            dataset,
            method=method,
            output=output,
            index=index,
            external=external,
            batch_size=batch_size,
        )
    if output is None and json:
        print(
            dumps(
                DatasetAnalyzer.analyze(result)
                .get_serializer()
                .serialize(result)
            )
        )


@mlem_command("apply-remote", section="runtime")
def apply_remote(
    subtype: str = Argument(
        "",
        help=f"Type of client. Choices: {list_implementations(BaseClient)}",
        show_default=False,
    ),
    data: str = Argument(..., help="Path to dataset object"),
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    target_repo: Optional[str] = option_target_repo,
    method: str = option_method,
    index: bool = option_index,
    json: bool = option_json,
    load: Optional[str] = option_load("client"),
    conf: List[str] = option_conf("client"),
    file_conf: List[str] = option_file_conf("client"),
):
    """Apply a model (deployed somewhere remotely) to a dataset. Resulting dataset will be saved as MLEM object to `output` if it is provided, otherwise will be printed

    Examples:
        Apply hosted mlem model to local mlem dataset
        $ mlem apply-remote http mydataset -c host="0.0.0.0" -c port=8080 --output myprediction
    """
    client = config_arg(BaseClient, load, subtype, conf, file_conf)

    with set_echo(None if json else ...):
        result = run_apply_remote(
            client, data, repo, rev, index, method, output, target_repo
        )
    if output is None and json:
        print(
            dumps(
                DatasetAnalyzer.analyze(result)
                .get_serializer()
                .serialize(result)
            )
        )


def run_apply_remote(
    client: BaseClient,
    data: str,
    repo,
    rev,
    index,
    method,
    output,
    target_repo,
):
    from mlem.api import apply_remote

    dataset = load_meta(
        data,
        repo,
        rev,
        load_value=True,
        force_type=MlemDataset,
    )
    result = apply_remote(
        client,
        dataset,
        method=method,
        output=output,
        target_repo=target_repo,
        index=index,
    )
    return result
