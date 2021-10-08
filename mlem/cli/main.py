import logging
from functools import wraps

import click

from mlem import version


@click.group()
@click.version_option(version.__version__)
def cli():
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialise any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment
    """


def _set_log_level(ctx, param, value):  # pylint: disable=unused-argument
    if value:
        logger = logging.getLogger("mlem")
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)


verbose_option = click.option(
    "-v",
    "--verbose",
    callback=_set_log_level,
    expose_value=False,
    is_eager=True,
    is_flag=True,
)


@wraps(cli.command)
def mlem_command(*args, **kwargs):
    def decorator(f):
        return cli.command(*args, **kwargs)(verbose_option(f))

    return decorator  # cli.command(*args, **kwargs)(verbose_option)
