from json import dumps
from typing import Optional

from typer import Argument, Option

from mlem.api import import_object
from mlem.cli.main import (
    mlem_command,
    option_external,
    option_json,
    option_link,
    option_repo,
    option_rev,
)
from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.dataset_type import DatasetAnalyzer
from mlem.core.import_objects import ImportHook
from mlem.core.metadata import load_meta
from mlem.core.objects import DatasetMeta, ModelMeta
from mlem.ext import list_implementations
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
    method: str = Option(
        PREDICT_METHOD_NAME,
        "-m",
        "--method",
        help="Which model method is to apply",
    ),
    data_repo: Optional[str] = Option(
        None,
        "--data-repo",
        "--dr",
        help="Repo with dataset",
    ),
    data_rev: Optional[str] = Option(
        None,
        "--data-rev",
        help="Revision of dataset",
    ),
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
    link: bool = option_link,
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
            dataset = import_object(
                data, repo=data_repo, rev=data_rev, type_=import_type
            )
        else:
            dataset = load_meta(
                data,
                data_repo,
                data_rev,
                load_value=True,
                force_type=DatasetMeta,
            )
        meta = load_meta(model, repo, rev, force_type=ModelMeta)

        result = apply(
            meta,
            dataset,
            method=method,
            output=output,
            link=link,
            external=external,
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
    host: str = Argument(
        default="0.0.0.0", help="Host address of where model is being served"
    ),
    port: Optional[int] = Argument(
        default=8080,
        help="Port for communicating with the server where model is hosted",
    ),
    data: str = Argument(..., help="Path to dataset object"),
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    method: str = Option(
        PREDICT_METHOD_NAME,
        "-m",
        "--method",
        help="Which model method is to apply",
    ),
    link: bool = option_link,
    json: bool = option_json,
):
    """Apply a model to a dataset. Resulting dataset will be saved as MLEM object to `output` if it is provided, otherwise will be printed

    Examples:
        Apply hosted mlem model to local mlem dataset
        $ mlem apply-remote "0.0.0.0" 8080 mydataset --method predict --output myprediction
    """
    from mlem.api import apply_remote

    with set_echo(None if json else ...):
        dataset = load_meta(
            data,
            None,
            None,
            load_value=True,
            force_type=DatasetMeta,
        )

        result = apply_remote(
            host,
            port,
            dataset,
            method=method,
            output=output,
            link=link,
        )
    if output is None and json:
        print(
            dumps(
                DatasetAnalyzer.analyze(result)
                .get_serializer()
                .serialize(result)
            )
        )
