import logging
from functools import wraps
from typing import List, Optional, Type

import click
import typer
from pydantic import parse_obj_as
from typer import Context, Option, Typer
from yaml import safe_load

from mlem import version
from mlem.analytics import send_cli_call
from mlem.constants import MLEM_DIR
from mlem.core.base import MlemObject, build_mlem_object
from mlem.core.errors import MlemError

app = Typer()


@app.callback(invoke_without_command=True, no_args_is_help=True)
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
        typer.echo(f"MLEM Version: {version.__version__}")
    if verbose:
        logger = logging.getLogger("mlem")
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    ctx.obj = {"traceback": traceback}


def mlem_command(*args, parent=app, **kwargs):
    def decorator(f):
        if len(args) > 0:
            cmd_name = args[0]
        else:
            cmd_name = kwargs.get("name", f.__name__)

        @parent.command(*args, **kwargs)
        @wraps(f)
        @click.pass_context
        def inner(ctx, *iargs, **ikwargs):
            res = {}
            error = None
            try:
                res = f(*iargs, **ikwargs) or {}
                res = {f"cmd_{cmd_name}_{k}": v for k, v in res.items()}
            except MlemError as e:
                error = str(type(e))
                if ctx.obj["traceback"]:
                    raise
                typer.echo(str(e), err=True, color=typer.colors.RED)
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


option_repo = Option(None, "-r", "--repo", help="Path to MLEM repo")
option_rev = Option(None, "--rev", help="Repo revision to use.")
option_link = Option(
    False,
    "--link/--no-link",
    help="Whether to create link for output in .mlem directory.",
)
option_external = Option(
    False,
    "--external",
    "-e",
    is_flag=True,
    help=f"Save object not in {MLEM_DIR}, but directly in repo.",
)
option_target_repo = Option(
    None,
    "--target-repo",
    "--tr",
    help="Save object to mlem dir found in {target_repo} path.",
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


#
#
# def config_arg(name: str, model: Type[MlemObject], **kwargs):
#     """add argument + multi option -c and -f to configure and deserialize to model"""
#
#     def decorator(f):
#
#
#
#         @wraps(f)
#         def inner(
#             load: Optional[str]=Option(None, "-l", "--load", ),
#             subtype: str=Argument(..., "subtype", default="", **kwargs),
#             conf: List[str]=Option(None, "-c", "--conf", multiple=True),
#             file_conf: List[str]=Option(None, "-f", "--file_conf", multiple=True),
#             **inner_kwargs,
#         ):
#             if load is not None:
#                 with open(load, "r", encoding="utf8") as of:
#                     obj = parse_obj_as(model, safe_load(of))
#             else:
#                 obj = build_mlem_object(model, subtype, conf, file_conf)
#             inner_kwargs[name] = obj
#             return f(**inner_kwargs)
#
#         return inner
#
#     return decorator
#
#
# def _get_option_name(option: Callable):
#     option_decls = option.__closure__[-1].cell_contents  # type: ignore
#     return option_decls[-1].lstrip("-")
#
#
# def with_meta(
#     argument_name: str,
#     repo_opt: Callable = option_repo,
#     rev_opt: Callable = option_rev,
#     force_cls: Type[MlemMeta] = None,
# ):
#     def decorator(f):
#         Argument(..., argument_name)
#         @repo_opt
#         @rev_opt
#         @wraps(f)
#         def inner(**kwargs):
#             path = kwargs.pop(argument_name)
#             repo = kwargs.pop(_get_option_name(repo_opt))
#             rev = kwargs.pop(_get_option_name(rev_opt))
#             meta = load_meta(path, repo=repo, rev=rev, force_type=force_cls)
#             kwargs[argument_name] = meta
#             return f(**kwargs)
#
#         return inner
#
#     return decorator
