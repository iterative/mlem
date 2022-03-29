from typing import Optional

import click
from typer import Option

from mlem.api import import_object
from mlem.cli.main import mlem_command, option_link, option_repo, option_rev
from mlem.core.metadata import load_meta
from mlem.core.objects import DatasetMeta, ModelMeta


@mlem_command("apply")
def apply(
    model: str,
    data: str,
    repo: Optional[str] = option_repo,
    rev: Optional[str] = option_rev,
    output: Optional[str] = Option(
        None, "-o", "--output", help="Where to store the outputs."
    ),
    method: str = Option(
        None,
        "-m",
        "--method",
        help="Which model method is to apply (if the model is instance of class).",
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
    ),
    link: bool = option_link,
):
    """Apply a model to supplied data."""
    from mlem.api import apply

    if import_:
        dataset = import_object(
            data, repo=data_repo, rev=data_rev, type_=import_type
        )
    else:
        dataset = load_meta(
            data, data_repo, data_rev, load_value=True, force_type=DatasetMeta
        )
    result = apply(
        load_meta(model, repo, rev, force_type=ModelMeta),
        dataset,
        method=method,
        output=output,
        link=link,
    )
    if output is None:
        click.echo(result)
