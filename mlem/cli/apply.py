from typing import Any, Tuple

import click

from mlem.cli.main import cli
from mlem.cli.utils import with_model_meta
from mlem.core.objects import ModelMeta


@cli.command("apply")
@with_model_meta
@click.option(
    "-o", "--output", default=None, help="Where to store the outputs."
)
@click.option(
    "-m",
    "--method",
    default=None,
    help="Which model method is to apply (if the model is instance of class).",
)
@click.argument("args", nargs=-1)
@click.option(
    "--link/--no-link",
    default=False,
    help="Whether to create links for outputs in .mlem directory.",
)
def apply(
    model: ModelMeta, output: str, method: str, args: Tuple[Any], link: bool
):
    """Apply a model to supplied data."""
    from mlem.api import apply

    click.echo("applying")
    apply(model, *args, method=method, output=output, link=link)


@cli.command()
@with_model_meta
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
