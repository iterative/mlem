from typing import Optional

import click

from mlem.api import import_object
from mlem.cli.main import mlem_command, option_link, with_meta
from mlem.core.metadata import load_meta
from mlem.core.objects import DatasetMeta, ModelMeta


@mlem_command("apply")
@with_meta("model", force_cls=ModelMeta)
@click.option(
    "-o", "--output", default=None, help="Where to store the outputs."
)
@click.option(
    "-m",
    "--method",
    default=None,
    help="Which model method is to apply (if the model is instance of class).",
)
@click.argument("data")
@click.option("--data-repo", "--dr", help="Repo with dataset", default=None)
@click.option("--data-rev", help="Revision of dataset", default=None)
@click.option(
    "-i",
    "--import",
    "import_",
    help="Try to import data on-the-fly",
    is_flag=True,
    default=False,
)
@click.option("--import-type", "--it", default=None)
@option_link
def apply(
    model: ModelMeta,
    output: Optional[str],
    method: str,
    data: str,
    data_repo: Optional[str],
    data_rev: Optional[str],
    import_: bool,
    import_type: str,
    link: bool,
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
    result = apply(model, dataset, method=method, output=output, link=link)
    if output is None:
        click.echo(result)


@mlem_command()
@with_meta("model", force_cls=ModelMeta)
@click.argument("output")
@click.option(
    "-m",
    "--method",
    default=None,
    help="Which model method is to apply (if the model is instance of class).",
)
@click.argument("args", nargs=-1)
def apply_remote(
    model: ModelMeta, output, method, args
):  # pylint: disable=unused-argument
    click.echo("applying remote")
    raise NotImplementedError()  # TODO: https://github.com/iterative/mlem/issues/30
    # if meta.is_data:
    #     click.echo('nothing to do here')
    #     return
    #
    # if meta.deployment is None:
    #     click.Abort(f'{meta} is not deployed')
    #     return
    #
    # client = meta.deployment.get_client()
    #
    # # method = meta.model.resolve_method(method)
    #
    # res = getattr(client, method)(load(args[0]))
    #
    # save(res, output)
