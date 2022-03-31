import logging
from functools import wraps
from typing import List, Optional, Tuple, Type

import click
import typer
from click import HelpFormatter
from pydantic import parse_obj_as
from typer import Context, Option, Typer
from typer.core import TyperCommand
from yaml import safe_load

from mlem import version
from mlem.analytics import send_cli_call
from mlem.constants import MLEM_DIR
from mlem.core.base import MlemObject, build_mlem_object
from mlem.core.errors import MlemError
from mlem.ui import EMOJI_FAIL, EMOJI_MLEM, bold, cli_echo, color, echo


def mlem_callback(
    ctx: Context,
    show_version: bool = Option(False, "--version"),
    verbose: bool = Option(False, "--verbose", "-v"),
    traceback: bool = Option(False, "--traceback", "--tb"),
):
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialize any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment
    """
    if ctx.invoked_subcommand is None and show_version:
        with cli_echo():
            echo(EMOJI_MLEM + f"MLEM Version: {version.__version__}")
    if verbose:
        logger = logging.getLogger("mlem")
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    ctx.obj = {"traceback": traceback}


def _extract_examples(
    help_str: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if help_str is None:
        return None, None
    try:
        examples = help_str.index("Examples:")
    except ValueError:
        return None, help_str
    return help_str[examples + len("Examples:") + 1 :], help_str[:examples]


class MlemFormatter(click.HelpFormatter):
    def write_heading(self, heading: str) -> None:
        super().write_heading(bold(heading))


class MlemCommand(TyperCommand):
    def __init__(
        self, name: Optional[str], help: Optional[str] = None, **kwargs
    ):
        self.examples, help = _extract_examples(help)
        super().__init__(name, help=help, **kwargs)

    def get_help(self, ctx: Context) -> str:
        """Formats the help into a string and returns it.

        Calls :meth:`format_help` internally.
        """
        formatter = MlemFormatter(
            width=ctx.terminal_width, max_width=ctx.max_content_width
        )
        self.format_help(ctx, formatter)
        return formatter.getvalue().rstrip("\n")

    def format_epilog(self, ctx: Context, formatter: HelpFormatter) -> None:
        super().format_epilog(ctx, formatter)
        if self.examples:
            with formatter.section("Examples"):
                formatter.write(self.examples)


app = Typer(
    no_args_is_help=True, callback=mlem_callback, invoke_without_command=True
)


def mlem_command(*args, parent=app, **kwargs):
    def decorator(f):
        if len(args) > 0:
            cmd_name = args[0]
        else:
            cmd_name = kwargs.get("name", f.__name__)

        @parent.command(*args, **kwargs, cls=MlemCommand)
        @wraps(f)
        @click.pass_context
        def inner(ctx, *iargs, **ikwargs):
            res = {}
            error = None
            try:
                with cli_echo():
                    res = f(*iargs, **ikwargs) or {}
                res = {f"cmd_{cmd_name}_{k}": v for k, v in res.items()}
            except MlemError as e:
                error = str(type(e))
                if ctx.obj["traceback"]:
                    raise
                with cli_echo():
                    echo(EMOJI_FAIL + color(str(e), col=typer.colors.RED))
                raise typer.Exit(1)
            except Exception as e:
                error = str(type(e))
                if ctx.obj["traceback"]:
                    raise
                raise e
            finally:
                send_cli_call(cmd_name, error_msg=error, **res)

        return inner

    return decorator


option_repo = Option(
    None, "-r", "--repo", help="Path to MLEM repo", show_default=True
)
option_rev = Option(None, "--rev", help="Repo revision to use")
option_link = Option(
    False,
    "--link/--no-link",
    help="Whether to create link for output in .mlem directory",
)
option_external = Option(
    False,
    "--external",
    "-e",
    is_flag=True,
    help=f"Save result not in {MLEM_DIR}, but directly in repo",
)
option_target_repo = Option(
    None,
    "--target-repo",
    "--tr",
    help="Repo to save target to",
)


def config_arg(
    model: Type[MlemObject],
    load: Optional[str],
    subtype: str,
    conf: Optional[List[str]],
    file_conf: Optional[List[str]],
):
    if load is not None:
        with open(load, "r", encoding="utf8") as of:
            obj = parse_obj_as(model, safe_load(of))
    else:
        obj = build_mlem_object(model, subtype, conf, file_conf)
    return obj
