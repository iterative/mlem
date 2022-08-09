import contextlib
import logging
import typing as t
from collections import defaultdict
from enum import Enum, EnumMeta
from functools import partial, wraps
from gettext import gettext
from typing import List, Optional, Tuple, Type

import typer
from click import (
    Abort,
    ClickException,
    Command,
    HelpFormatter,
    Parameter,
    pass_context,
)
from click.exceptions import Exit
from pydantic import BaseModel, MissingError, ValidationError, parse_obj_as
from pydantic.error_wrappers import ErrorWrapper
from pydantic.typing import get_args
from simple_parsing.docstring import get_attribute_docstring
from typer import Context, Option, Typer
from typer.core import TyperCommand, TyperGroup, TyperOption
from yaml import safe_load

from mlem import LOCAL_CONFIG, version
from mlem.constants import MLEM_DIR, PREDICT_METHOD_NAME
from mlem.core.base import MlemABC, build_mlem_object
from mlem.core.errors import MlemError
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemObject
from mlem.telemetry import telemetry
from mlem.ui import EMOJI_FAIL, EMOJI_MLEM, bold, cli_echo, color, echo
from mlem.utils.entrypoints import list_implementations


class MlemFormatter(HelpFormatter):
    def write_heading(self, heading: str) -> None:
        super().write_heading(bold(heading))


class MlemMixin(Command):
    def __init__(
        self,
        *args,
        examples: Optional[str],
        section: str = "other",
        aliases: List[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.examples = examples
        self.section = section
        self.aliases = aliases
        self.rich_help_panel = section.capitalize()

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
        dynamic_options_generator: t.Callable[
            [], t.Iterable[Parameter]
        ] = None,
        dynamic_metavar: str = None,
        **kwargs,
    ):
        self.dynamic_metavar = dynamic_metavar
        self.dynamic_options_generator = dynamic_options_generator
        examples, help = _extract_examples(help)
        super().__init__(
            name=name,
            section=section,
            aliases=aliases,
            examples=examples,
            help=help,
            **kwargs,
        )

    def get_params(self, ctx) -> List["Parameter"]:
        res: List[Parameter] = (
            list(self.dynamic_options_generator())
            if self.dynamic_options_generator is not None
            else []
        )
        res = res + super().get_params(ctx)
        if self.dynamic_metavar is not None:
            kw_param = [p for p in res if p.name == self.dynamic_metavar]
            if len(kw_param) > 0:
                res.remove(kw_param[0])
        return res


class MlemGroup(MlemMixin, TyperGroup):
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
            name=name,
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


def mlem_group(section, aliases: Optional[List[str]] = None):
    class MlemGroupSection(MlemGroup):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, section=section, aliases=aliases, **kwargs)

    return MlemGroupSection


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


class CliTypeField(BaseModel):
    required: bool
    path: str
    type_: Type
    help: str
    default: t.Any

    def to_text(self):
        req = (
            color("[required]", "")
            if self.required
            else color("[not required]", "white")
        )
        if not self.required:
            default = self.default
            if isinstance(default, str):
                default = f'"{default}"'
            default = f" = {default}"
        else:
            default = ""
        return (
            req
            + " "
            + color(self.path, "green")
            + ": "
            + str(self.type_.__name__)
            + default
            + "\n\t"
            + self.help
        )


def get_field_help(cls: Type, field_name: str):
    return (
        get_attribute_docstring(cls, field_name).docstring_below
        or "Field docstring missing"
    )


def iterate_type_fields(cls: Type[BaseModel], prefix="", force_not_req=False):
    for name, field in sorted(
        cls.__fields__.items(), key=lambda x: not x[1].required
    ):
        name = field.alias or name
        if issubclass(cls, MlemObject) and name in MlemObject.__fields__:
            continue
        if issubclass(cls, MlemABC) and name in cls.__config__.exclude:
            continue
        fullname = name if not prefix else f"{prefix}.{name}"

        req = field.required and not force_not_req
        default = field.default
        docstring = get_field_help(cls, name)
        field_type = field.type_
        if not isinstance(field_type, type):
            # typing.GenericAlias
            generic_args = get_args(field_type)
            if len(generic_args) > 0:
                field_type = field_type.__args__[0]
            else:
                field_type = object
        if (
            isinstance(field_type, type)
            and issubclass(field_type, MlemABC)
            and field_type.__is_root__
        ):
            if isinstance(default, field_type):
                default = default.__get_alias__()
            yield CliTypeField(
                required=req,
                path=fullname,
                type_=str,
                help=f"{docstring}. One of {list_implementations(field_type)}. Run 'mlem types {field_type.abs_name} <subtype>' for list of nested fields for each subtype",
                default=default,
            )
        elif isinstance(field_type, type) and issubclass(
            field_type, BaseModel
        ):
            yield from iterate_type_fields(
                field_type, fullname, not field.required
            )
        else:
            yield CliTypeField(
                required=req,
                path=fullname,
                type_=field_type,
                default=default,
                help=docstring,
            )


def abc_fields_parameters(cls: Type[MlemABC]):
    def generator():
        for field in iterate_type_fields(cls):
            option = TyperOption(
                param_decls=[f"--{field.path}", field.path.replace(".", "_")],
                type=field.type_,
                required=field.required,
                default=field.default,
                help=field.help,
                show_default=True,
            )
            option.name = field.path
            yield option

    return generator


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
):
    """\b
    MLEM is a tool to help you version and deploy your Machine Learning models:
    * Serialize any model trained in Python into ready-to-deploy format
    * Model lifecycle management using Git and GitOps principles
    * Provider-agnostic deployment

    Examples:
        $ mlem init
        $ mlem list https://github.com/iterative/example-mlem
        $ mlem clone models/logreg --project https://github.com/iterative/example-mlem --rev main logreg
        $ mlem link logreg latest
        $ mlem apply latest https://github.com/iterative/example-mlem/data/test_x -o pred
        $ mlem serve latest fastapi -c port=8001
        $ mlem build latest docker_dir -c target=build/ -c server.type=fastapi
    """
    if ctx.invoked_subcommand is None and show_version:
        with cli_echo():
            echo(EMOJI_MLEM + f"MLEM Version: {version.__version__}")
    if verbose:
        logger = logging.getLogger("mlem")
        logger.handlers[0].setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    ctx.obj = {"traceback": traceback or LOCAL_CONFIG.DEBUG}


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
    mlem_cls=None,
    dynamic_metavar=None,
    dynamic_options_generator=None,
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
            cls=partial(
                mlem_cls or MlemCommand,
                section=section,
                aliases=aliases,
                dynamic_options_generator=dynamic_options_generator,
                dynamic_metavar=dynamic_metavar,
            ),
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
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                raise
            except MlemError as e:
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                if ctx.obj["traceback"]:
                    raise
                with cli_echo():
                    echo(EMOJI_FAIL + color(str(e), col=typer.colors.RED))
                raise typer.Exit(1)
            except ValidationError as e:
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
                if ctx.obj["traceback"]:
                    raise
                msgs = "\n".join(_format_validation_error(e))
                with cli_echo():
                    echo(msgs)
                raise typer.Exit(1)
            except Exception as e:  # pylint: disable=broad-except
                error = f"{e.__class__.__module__}.{e.__class__.__name__}"
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
                telemetry.send_cli_call(cmd_name, error=error, **res)

        return inner

    return decorator


option_project = Option(
    None, "-p", "--project", help="Path to MLEM project", show_default="none"  # type: ignore
)
option_method = Option(
    PREDICT_METHOD_NAME,
    "-m",
    "--method",
    help="Which model method is to be applied",
)
option_rev = Option(None, "--rev", help="Repo revision to use", show_default="none")  # type: ignore
option_index = Option(
    None,
    "--index/--no-index",
    help="Whether to index output in .mlem directory",
)
option_external = Option(
    None,
    "--external",
    "-e",
    is_flag=True,
    help=f"Save result not in {MLEM_DIR}, but directly in project",
)
option_target_project = Option(
    None,
    "--target-project",
    "--tp",
    help="Project to save target to",
    show_default="none",  # type: ignore
)
option_json = Option(False, "--json", help="Output as json")
option_data_project = Option(
    None,
    "--data-project",
    "--dr",
    help="Project with data",
)
option_data_rev = Option(
    None,
    "--data-rev",
    help="Revision of data",
)


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
def wrap_build_error(subtype, model: Type[MlemABC]):
    try:
        yield
    except ValidationError as e:
        msgs = "\n".join(_format_validation_error(e))
        raise typer.BadParameter(
            f"Error on constructing {subtype} {model.abs_name}:\n{msgs}"
        ) from e


def config_arg(
    model: Type[MlemABC],
    load: Optional[str],
    subtype: str,
    conf: Optional[List[str]],
    file_conf: Optional[List[str]],
    **kwargs,
):
    obj: MlemABC
    if load is not None:
        if issubclass(model, MlemObject):
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
            obj = build_mlem_object(model, subtype, conf, file_conf, kwargs)

    return obj
