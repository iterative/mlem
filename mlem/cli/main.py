import inspect
import logging
from collections import defaultdict
from functools import partial, wraps
from gettext import gettext
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import click
import typer
from click import Abort, ClickException, Command, HelpFormatter, Parameter
from click.exceptions import Exit, MissingParameter, NoSuchOption
from pydantic import ValidationError
from typer import Context, Option, Typer
from typer.core import TyperCommand, TyperGroup

from mlem import LOCAL_CONFIG, version
from mlem.cli.utils import (
    FILE_CONF_PARAM_NAME,
    LOAD_PARAM_NAME,
    NOT_SET,
    CallContext,
    _format_validation_error,
    get_extra_keys,
)
from mlem.constants import PREDICT_METHOD_NAME
from mlem.core.errors import MlemError
from mlem.telemetry import telemetry
from mlem.ui import (
    EMOJI_FAIL,
    EMOJI_MLEM,
    bold,
    cli_echo,
    color,
    echo,
    no_echo,
    stderr_echo,
)

PATH_METAVAR = "path"
COMMITISH_METAVAR = "commitish"
TRACEBACK_SUGGESTION_MESSAGE = (
    "Use the --tb or --traceback option to include the traceback in the output"
)


class MlemFormatter(HelpFormatter):
    def write_heading(self, heading: str) -> None:
        super().write_heading(bold(heading))


class MlemMixin(Command):
    def __init__(
        self,
        *args,
        section: str = "other",
        aliases: List[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.section = section
        self.aliases = aliases
        self.rich_help_panel = section.capitalize()

    def collect_usage_pieces(self, ctx: Context) -> List[str]:
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

    def _get_cmd_name_for_docs_link(self):
        ctx = click.get_current_context()
        return get_cmd_name(ctx, no_aliases=True, sep="/")

    @staticmethod
    def _add_docs_link(help, cmd_name):
        return (
            help
            if help is None or "Documentation" in help
            else f"{help}\n\nDocumentation: <https://mlem.ai/doc/command-reference/{cmd_name}>"
        )


class MlemCommand(
    MlemMixin,
    TyperCommand,
):
    def __init__(
        self,
        name: Optional[str],
        section: str = "other",
        aliases: List[str] = None,
        help: Optional[str] = None,
        dynamic_options_generator: Callable[
            [CallContext], Iterable[Parameter]
        ] = None,
        dynamic_metavar: str = None,
        lazy_help: Optional[Callable[[], str]] = None,
        pass_from_parent: Optional[List[str]] = None,
        **kwargs,
    ):
        self.dynamic_metavar = dynamic_metavar
        self.dynamic_options_generator = dynamic_options_generator
        self._help = help
        self.lazy_help = lazy_help
        self.pass_from_parent = pass_from_parent
        super().__init__(
            name=name,
            section=section,
            aliases=aliases,
            help=help,
            **kwargs,
        )

    def make_context(
        self,
        info_name: Optional[str],
        args: List[str],
        parent: Optional[Context] = None,
        **extra: Any,
    ) -> Context:
        args_copy = args[:]
        if self.dynamic_options_generator:
            extra["ignore_unknown_options"] = True
        ctx = super().make_context(info_name, args, parent, **extra)
        if not self.dynamic_options_generator:
            return ctx
        extra_args = ctx.args
        params = ctx.params.copy()
        while extra_args:
            ctx.params = params
            ctx.args = args_copy[:]
            with ctx.scope(cleanup=False):
                self.parse_args(ctx, args_copy[:])
                params.update(ctx.params)

            if ctx.args == extra_args:
                if not self.ignore_unknown_options:
                    from difflib import get_close_matches

                    opt = "--" + get_extra_keys(extra_args)[0]
                    possibilities = get_close_matches(
                        opt, {f"--{o}" for o in params}
                    )
                    raise NoSuchOption(
                        opt, possibilities=possibilities, ctx=ctx
                    )
                break
            extra_args = ctx.args

        return ctx

    def invoke(self, ctx: Context) -> Any:
        ctx.params = {k: v for k, v in ctx.params.items() if v != NOT_SET}
        return super().invoke(ctx)

    def get_params(self, ctx) -> List["Parameter"]:
        regular_options = super().get_params(ctx)
        res: List[Parameter] = (
            list(
                self.dynamic_options_generator(
                    CallContext(
                        ctx.params,
                        get_extra_keys(ctx.args),
                        [o.name for o in regular_options],
                    )
                )
            )
            if self.dynamic_options_generator is not None
            else []
        ) + regular_options

        if self.dynamic_metavar is not None:
            kw_param = [p for p in res if p.name == self.dynamic_metavar]
            if len(kw_param) > 0:
                res.remove(kw_param[0])
        if self.pass_from_parent is not None:
            res = [
                o
                for o in res
                if o.name not in self.pass_from_parent
                or o.name not in ctx.parent.params
                or ctx.parent.params[o.name] is None
            ]
        return res

    @property
    def help(self):
        cmd_name = self._get_cmd_name_for_docs_link()
        if self.lazy_help:
            if "/" in cmd_name:
                cmd_name = cmd_name[: cmd_name.index("/")]
            return self._add_docs_link(self.lazy_help(), cmd_name)
        return self._add_docs_link(self._help, cmd_name)

    @help.setter
    def help(self, value):
        self._help = value


class MlemGroup(MlemMixin, TyperGroup):
    order = ["common", "object", "runtime", "other"]

    def __init__(
        self,
        name: Optional[str] = None,
        commands: Optional[
            Union[Dict[str, Command], Sequence[Command]]
        ] = None,
        section: str = "other",
        aliases: List[str] = None,
        help: str = None,
        **attrs: Any,
    ) -> None:
        super().__init__(
            name=name,
            help=help,
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

    def get_command(self, ctx: Context, cmd_name: str) -> Optional[Command]:
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

    @property
    def help(self):
        cmd_name = self._get_cmd_name_for_docs_link()
        if "/" in cmd_name:
            cmd_name = cmd_name[: cmd_name.index("/")]
        return self._add_docs_link(self._help, cmd_name)

    @help.setter
    def help(self, value):
        self._help = value


def mlem_group(section, aliases: Optional[List[str]] = None):
    class MlemGroupSection(MlemGroup):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, section=section, aliases=aliases, **kwargs)

    return MlemGroupSection


def mlem_group_callback(group: Typer, required: Optional[List[str]] = None):
    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            ctx = click.get_current_context()
            if ctx.invoked_subcommand is not None:
                return None
            if required is not None:
                for req in required:
                    if req not in kwargs or kwargs[req] is None:
                        param = [
                            p
                            for p in ctx.command.get_params(ctx)
                            if p.name == req
                        ][0]
                        raise MissingParameter(ctx=ctx, param=param)
            return f(*args, **kwargs)

        return group.callback(invoke_without_command=True)(
            wrap_mlem_cli_call(inner, None)
        )

    return decorator


app = Typer(
    cls=MlemGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
# available from typer>=0.6
app.pretty_exceptions_enable = False
app.pretty_exceptions_show_locals = False


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
    quiet: bool = Option(False, "--quiet", "-q", help="Suppress output"),
):
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialize any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment
    \b
    Documentation: <https://mlem.ai/doc>
    """
    if ctx.invoked_subcommand is None and show_version:
        with cli_echo():
            echo(EMOJI_MLEM + f"MLEM Version: {version.__version__}")
    if verbose:
        logger = logging.getLogger("mlem")
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    ctx.obj = {
        "traceback": traceback or verbose or LOCAL_CONFIG.DEBUG,
        "quiet": quiet,
    }


def get_cmd_name(ctx: Context, no_aliases=False, sep=" "):
    pieces = []
    while ctx.parent is not None:
        pieces.append(ctx.command.name if no_aliases else ctx.info_name)
        ctx = ctx.parent
    return sep.join(reversed(pieces))


def mlem_command(
    *args,
    section="other",
    aliases=None,
    options_metavar="[options]",
    parent=app,
    mlem_cls=None,
    dynamic_metavar=None,
    dynamic_options_generator=None,
    lazy_help=None,
    pass_from_parent: Optional[List[str]] = None,
    no_pass_from_parent: Optional[List[str]] = None,
    **kwargs,
):
    def decorator(f):
        context_settings = kwargs.get("context_settings", {})
        if dynamic_options_generator:
            context_settings.update({"allow_extra_args": True})
        if no_pass_from_parent is not None:
            _pass_from_parent = [
                a
                for a in inspect.getfullargspec(f).args
                if a not in no_pass_from_parent
            ]
        else:
            _pass_from_parent = pass_from_parent
        call = wrap_mlem_cli_call(f, _pass_from_parent)
        return parent.command(
            *args,
            options_metavar=options_metavar,
            context_settings=context_settings,
            **kwargs,
            cls=partial(
                mlem_cls or MlemCommand,
                section=section,
                aliases=aliases,
                dynamic_options_generator=dynamic_options_generator,
                dynamic_metavar=dynamic_metavar,
                lazy_help=lazy_help,
                pass_from_parent=pass_from_parent,
            ),
        )(call)

    return decorator


def wrap_mlem_cli_call(f, pass_from_parent: Optional[List[str]]):
    @wraps(f)
    def inner(*iargs, **ikwargs):
        res = {}
        error = None
        ctx = click.get_current_context()
        cmd_name = get_cmd_name(ctx)
        try:
            if pass_from_parent is not None:
                ikwargs.update(
                    {
                        o: ctx.parent.params[o]
                        for o in pass_from_parent
                        if o in ctx.parent.params
                        and (o not in ikwargs or ikwargs[o] is None)
                    }
                )
            with (cli_echo() if not ctx.obj["quiet"] else no_echo()):
                res = f(*iargs, **ikwargs) or {}
            res = {f"cmd_{cmd_name}_{k}": v for k, v in res.items()}
        except (ClickException, Exit, Abort) as e:
            error = f"{e.__class__.__module__}.{e.__class__.__name__}"
            raise
        except MlemError as e:
            error = f"{e.__class__.__module__}.{e.__class__.__name__}"
            if ctx.obj["traceback"]:
                raise
            with stderr_echo():
                echo(EMOJI_FAIL + color(str(e), col=typer.colors.RED))
            raise typer.Exit(1)
        except ValidationError as e:
            error = f"{e.__class__.__module__}.{e.__class__.__name__}"
            if ctx.obj["traceback"]:
                raise
            msgs = "\n".join(_format_validation_error(e))
            with stderr_echo():
                echo(EMOJI_FAIL + color("Error:\n", "red") + msgs)
            raise typer.Exit(1)
        except Exception as e:  # pylint: disable=broad-except
            error = f"{e.__class__.__module__}.{e.__class__.__name__}"
            if ctx.obj["traceback"]:
                raise
            with stderr_echo():
                echo(
                    EMOJI_FAIL
                    + color(
                        "Unexpected error: " + str(e), col=typer.colors.RED
                    )
                )
                echo(TRACEBACK_SUGGESTION_MESSAGE)
                echo(
                    "Please report it here: <https://github.com/iterative/mlem/issues>"
                )
            raise typer.Exit(1)
        finally:
            if error is not None or ctx.invoked_subcommand is None:
                telemetry.send_cli_call(cmd_name, error=error, **res)

    return inner


option_project = Option(
    None,
    "-p",
    "--project",
    help="Path to MLEM project",
    metavar=PATH_METAVAR,
    show_default="none",  # type: ignore
)
option_method = Option(
    PREDICT_METHOD_NAME,
    "-m",
    "--method",
    help="Which model method is to be applied",
)
option_rev = Option(None, "--rev", help="Repo revision to use", show_default="none", metavar=COMMITISH_METAVAR)  # type: ignore
option_target_project = Option(
    None,
    "--target-project",
    "--tp",
    help="Project to save target to",
    metavar=PATH_METAVAR,
    show_default="none",  # type: ignore
)
option_json = Option(False, "--json", help="Output as json")
option_data_project = Option(
    None,
    "--data-project",
    "--dp",
    metavar=PATH_METAVAR,
    help="Project with data",
)
option_data_rev = Option(
    None,
    "--data-rev",
    "--dr",
    help="Revision of data",
    metavar=COMMITISH_METAVAR,
)
option_model_project = Option(
    None,
    "--model-project",
    "--mp",
    metavar=PATH_METAVAR,
    help="Project with model",
)
option_model_rev = Option(
    None,
    "--model-rev",
    "--mr",
    help="Revision of model",
    metavar=COMMITISH_METAVAR,
)
option_model = Option(
    ...,
    "-m",
    "--model",
    help="Path to MLEM model",
    metavar=PATH_METAVAR,
)
option_data = Option(
    ..., "-d", "--data", help="Path to MLEM dataset", metavar=PATH_METAVAR
)


def option_load(type_: str = None):
    type_ = type_ + " " if type_ is not None else ""
    return Option(
        None,
        "-l",
        f"--{LOAD_PARAM_NAME}",
        help=f"File to load {type_}config from",
        metavar=PATH_METAVAR,
    )


def option_file_conf(type_: str = None):
    type_ = f"for {type_} " if type_ is not None else ""
    return Option(
        None,
        "-f",
        f"--{FILE_CONF_PARAM_NAME}",
        help=f"File with options {type_}in format `field.name=path_to_config`",
    )
