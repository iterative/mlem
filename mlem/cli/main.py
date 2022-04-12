import contextlib
import logging
import typing as t
from collections import defaultdict
from enum import Enum, EnumMeta
from functools import partial, wraps
from gettext import gettext
from typing import List, Optional, Tuple, Type

import typer
from click import Abort, ClickException, Command, HelpFormatter, pass_context
from click.exceptions import Exit
from pydantic import BaseModel, MissingError, ValidationError, parse_obj_as
from pydantic.error_wrappers import ErrorWrapper
from typer import Context, Option, Typer
from typer.core import TyperCommand, TyperGroup
from yaml import safe_load

from mlem import version
from mlem.analytics import send_cli_call
from mlem.constants import MLEM_DIR
from mlem.core.base import MlemObject, build_mlem_object
from mlem.core.errors import MlemError
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemMeta
from mlem.ui import EMOJI_FAIL, EMOJI_MLEM, bold, cli_echo, color, echo


class MlemFormatter(HelpFormatter):
    def write_heading(self, heading: str) -> None:
        super().write_heading(bold(heading))


class MlemMixin(Command):
    def __init__(
        self,
        name: t.Optional[str],
        examples: Optional[str],
        section: str = "other",
        aliases: List[str] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.examples = examples
        self.section = section
        self.aliases = aliases

    def collect_usage_pieces(self, ctx: Context) -> t.List[str]:
        return [p.lower() for p in super().collect_usage_pieces(ctx)]

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


class MlemCommand(TyperCommand, MlemMixin):
    def __init__(
        self,
        name: Optional[str],
        section: str = "other",
        aliases: List[str] = None,
        help: Optional[str] = None,
        **kwargs,
    ):
        examples, help = _extract_examples(help)
        super().__init__(
            name,
            section=section,
            aliases=aliases,
            examples=examples,
            help=help,
            **kwargs,
        )


class MlemGroup(TyperGroup, MlemMixin):
    order = ["common", "object", "runtime", "other"]

    def __init__(
        self,
        name: t.Optional[str] = None,
        commands: t.Optional[
            t.Union[t.Dict[str, Command], t.Sequence[Command]]
        ] = None,
        section: str = "other",
        aliases: List[str] = None,
        help: str = None,
        **attrs: t.Any,
    ) -> None:
        examples, help = _extract_examples(help)
        super().__init__(
            name,
            help=help,
            examples=examples,
            aliases=aliases,
            section=section,
            commands=commands,
            **attrs,
        )

    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        commands = []
        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            # What is this, the tool lied about a command.  Ignore it
            if cmd is None:
                continue
            if cmd.hidden:
                continue

            commands.append((subcommand, cmd))

        # allow for 3 times the default spacing
        if len(commands) > 0:
            limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)

            sections = defaultdict(list)
            for subcommand, cmd in commands:
                help = cmd.get_short_help_str(limit)
                if isinstance(cmd, (MlemCommand, MlemGroup)):
                    section = cmd.section
                    aliases = (
                        f" ({','.join(cmd.aliases)})" if cmd.aliases else ""
                    )
                else:
                    section = "other"
                    aliases = ""

                sections[section].append((subcommand + aliases, help))

            for section in self.order:
                if sections[section]:
                    with formatter.section(
                        gettext(f"{section} commands".capitalize())
                    ):
                        formatter.write_dl(sections[section])

    def get_command(self, ctx: Context, cmd_name: str) -> t.Optional[Command]:
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if (
                isinstance(cmd, (MlemCommand, MlemGroup))
                and cmd.aliases
                and cmd_name in cmd.aliases
            ):
                return cmd
        return None


def MlemGroupSection(section, options_metavar="options"):
    return partial(MlemGroup, section=section, options_metavar=options_metavar)


class ChoicesMeta(EnumMeta):
    def __call__(cls, *names, module=None, qualname=None, type=None, start=1):
        if len(names) == 1:
            return super().__call__(names[0])
        return super().__call__(
            "Choice",
            names,
            module=module,
            qualname=qualname,
            type=type,
            start=start,
        )


class Choices(str, Enum, metaclass=ChoicesMeta):
    def _generate_next_value_(  # pylint: disable=no-self-argument
        name, start, count, last_values
    ):
        return name


app = Typer(
    cls=MlemGroup, context_settings={"help_option_names": ["-h", "--help"]}
)


@app.callback(no_args_is_help=True, invoke_without_command=True)
def mlem_callback(
    ctx: Context,
    show_version: bool = Option(
        False, "--version", help="Show version and exit"
    ),
    verbose: bool = Option(
        False, "--verbose", "-v", help="Print debug messages"
    ),
    traceback: bool = Option(False, "--traceback", "--tb", hidden=True),
):
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialize any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment

    Examples:
        $ mlem init
        $ mlem list https://github.com/iterative/example-mlem
        $ mlem clone models/logreg --repo https://github.com/iterative/example-mlem --rev main logreg
        $ mlem link logreg latest
        $ mlem apply latest https://github.com/iterative/example-mlem/data/test_x -o pred
        $ mlem serve latest fastapi -c port=8001
        $ mlem pack latest docker_dir -c target=build/ -c server.type=fastapi
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


def mlem_command(
    *args,
    section="other",
    aliases=None,
    options_metavar="[options]",
    parent=app,
    **kwargs,
):
    def decorator(f):
        if len(args) > 0:
            cmd_name = args[0]
        else:
            cmd_name = kwargs.get("name", f.__name__)

        @parent.command(
            *args,
            options_metavar=options_metavar,
            **kwargs,
            cls=partial(MlemCommand, section=section, aliases=aliases),
        )
        @wraps(f)
        @pass_context
        def inner(ctx, *iargs, **ikwargs):
            res = {}
            error = None
            try:
                with cli_echo():
                    res = f(*iargs, **ikwargs) or {}
                res = {f"cmd_{cmd_name}_{k}": v for k, v in res.items()}
            except (ClickException, Exit, Abort) as e:
                error = str(type(e))
                raise
            except MlemError as e:
                error = str(type(e))
                if ctx.obj["traceback"]:
                    raise
                with cli_echo():
                    echo(EMOJI_FAIL + color(str(e), col=typer.colors.RED))
                raise typer.Exit(1)
            except Exception as e:  # pylint: disable=broad-except
                error = str(type(e))
                if ctx.obj["traceback"]:
                    raise
                with cli_echo():
                    echo(
                        EMOJI_FAIL
                        + color(
                            "Unexpected error: " + str(e), col=typer.colors.RED
                        )
                    )
                    echo(
                        "Please report it here: <https://github.com/iterative/mlem/issues>"
                    )
                raise typer.Exit(1)
            finally:
                send_cli_call(cmd_name, error_msg=error, **res)

        return inner

    return decorator


option_repo = Option(
    None, "-r", "--repo", help="Path to MLEM repo", show_default="none"  # type: ignore
)
option_rev = Option(None, "--rev", help="Repo revision to use", show_default="none")  # type: ignore
option_link = Option(
    None,
    "--link/--no-link",
    help="Whether to create link for output in .mlem directory",
)
option_external = Option(
    None,
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
    show_default="none",  # type: ignore
)
option_json = Option(False, "--json", help="Output as json")


def option_load(type_: str = None):
    type_ = type_ + " " if type_ is not None else ""
    return Option(
        None, "-l", "--load", help=f"File to load {type_}config from"
    )


def option_conf(type_: str = None):
    type_ = f"for {type_} " if type_ is not None else ""
    return Option(
        None,
        "-c",
        "--conf",
        help=f"Options {type_}in format `field.name=value`",
    )


def option_file_conf(type_: str = None):
    type_ = f"for {type_} " if type_ is not None else ""
    return Option(
        None,
        "-f",
        "--file_conf",
        help=f"File with options {type_}in format `field.name=path_to_config`",
    )


def _iter_errors(
    errors: t.Sequence[t.Any], model: Type, loc: Optional[Tuple] = None
):
    for error in errors:
        if isinstance(error, ErrorWrapper):

            if loc:
                error_loc = loc + error.loc_tuple()
            else:
                error_loc = error.loc_tuple()

            if isinstance(error.exc, ValidationError):
                yield from _iter_errors(
                    error.exc.raw_errors, error.exc.model, error_loc
                )
            else:
                yield error_loc, model, error.exc


def _format_validation_error(error: ValidationError) -> List[str]:
    res = []
    for loc, model, exc in _iter_errors(error.raw_errors, error.model):
        path = ".".join(loc_part for loc_part in loc if loc_part != "__root__")
        field_type = model.__fields__[loc[-1]].type_
        if (
            isinstance(exc, MissingError)
            and isinstance(field_type, type)
            and issubclass(field_type, BaseModel)
        ):
            msgs = [
                str(EMOJI_FAIL + f"field `{path}.{f.name}`: {exc}")
                for f in field_type.__fields__.values()
                if f.required
            ]
            if msgs:
                res.extend(msgs)
            else:
                res.append(str(EMOJI_FAIL + f"field `{path}`: {exc}"))
        else:
            res.append(str(EMOJI_FAIL + f"field `{path}`: {exc}"))
    return res


@contextlib.contextmanager
def wrap_build_error(subtype, model: Type[MlemObject]):
    try:
        yield
    except ValidationError as e:
        msgs = "\n".join(_format_validation_error(e))
        raise typer.BadParameter(
            f"Error on constructing {subtype} {model.abs_name}:\n{msgs}"
        ) from e


def config_arg(
    model: Type[MlemObject],
    load: Optional[str],
    subtype: str,
    conf: Optional[List[str]],
    file_conf: Optional[List[str]],
):
    obj: MlemObject
    if load is not None:
        if issubclass(model, MlemMeta):
            obj = load_meta(load, force_type=model)
        else:
            with open(load, "r", encoding="utf8") as of:
                obj = parse_obj_as(model, safe_load(of))
    else:
        if not subtype:
            raise typer.BadParameter(
                f"Cannot configure {model.abs_name}: either subtype or --load should be provided"
            )
        with wrap_build_error(subtype, model):
            obj = build_mlem_object(model, subtype, conf, file_conf)

    return obj
