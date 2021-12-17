import logging
from functools import wraps

import click

from mlem import version
from mlem.analytics import send_cli_call
from mlem.constants import MLEM_DIR


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


def _send_analytics(cmd_name):
    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            res = {}
            error = None
            try:
                res = f(*args, **kwargs) or {}
                res = {f"cmd_{cmd_name}_{k}": v for k, v in res.items()}
            except Exception as e:
                error = str(type(e))
                raise e
            finally:
                send_cli_call(cmd_name, error_msg=error, **res)

        return inner

    return decorator


@wraps(cli.command)
def mlem_command(*args, **kwargs):
    def decorator(f):
        if len(args) > 0:
            cmd_name = args[0]
        else:
            cmd_name = kwargs.get("name", f.__name__)
        return cli.command(*args, **kwargs)(
            _send_analytics(cmd_name)(verbose_option(f))
        )

    return decorator


option_repo = click.option(
    "-r", "--repo", default=None, help="Path to MLEM repo"
)
option_rev = click.option("--rev", default=None, help="Repo revision to use.")
option_link = click.option(
    "--link/--no-link",
    default=False,
    help="Whether to create link for output in .mlem directory.",
)
option_external = click.option(
    "--external",
    "-e",
    default=False,
    is_flag=True,
    help=f"Save object not in {MLEM_DIR}, but directly in repo.",
)
option_target_repo = click.option(
    "--target-repo",
    "--tr",
    default=None,
    help="Save object to mlem dir found in {target_repo} path.",
)
