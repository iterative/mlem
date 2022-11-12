import pytest
import requests
from click import Context, Group
from typer.main import get_command_from_info, get_group, get_group_from_info

from mlem.cli import app
from tests.cli.conftest import Runner
from tests.conftest import long


def iter_group(group: Group, prefix=()):
    prefix = prefix + (group.name,)
    yield prefix, group
    for name, c in group.commands.items():
        if isinstance(c, Group):
            yield from iter_group(c, prefix)
        else:
            yield prefix + (name,), c


@pytest.fixture
def app_cli_cmd():
    commands = [
        get_command_from_info(
            c, pretty_exceptions_short=False, rich_markup_mode="rich"
        )
        for c in app.registered_commands
    ]
    groups = [
        get_group_from_info(
            g, pretty_exceptions_short=False, rich_markup_mode="rich"
        )
        for g in app.registered_groups
    ]
    return [(c.name, c) for c in commands] + [
        (" ".join(names), cmd) for g in groups for names, cmd in iter_group(g)
    ]


@long
def test_commands_help(app_cli_cmd):
    no_help = []
    no_link = []
    link_broken = []
    group = get_group(app)
    ctx = Context(group, info_name="mlem", help_option_names=["-h", "--help"])

    with ctx:
        for name, cli_cmd in app_cli_cmd:
            if cli_cmd.help is None:
                no_help.append(name)
            elif "Documentation: <" not in cli_cmd.help:
                no_link.append(name)
            else:
                link = cli_cmd.help.split("Documentation: <")[1].split(">")[0]
                response = requests.head(link, timeout=5)
                try:
                    response.raise_for_status()
                except requests.HTTPError:
                    link_broken.append(name)

    assert len(no_help) == 0, f"{no_help} cli commands do not have help!"
    assert (
        len(no_link) == 0
    ), f"{no_link} cli commands do not have documentation link!"
    assert (
        len(link_broken) == 0
    ), f"{link_broken} cli commands have broken documentation links!"


def test_commands_args_help(app_cli_cmd):
    no_help = []
    for name, cmd in app_cli_cmd:
        dynamic_metavar = getattr(cmd, "dynamic_metavar", None)
        for arg in cmd.params:
            if arg.name == dynamic_metavar:
                continue
            if arg.help is None:
                no_help.append(f"{name}:{arg.name}")
    assert len(no_help) == 0, f"{no_help} cli commands args do not have help!"


@pytest.mark.parametrize("cmd", ["--help", "-h"])
def test_help(runner: Runner, cmd):
    result = runner.invoke(cmd)
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )


def test_cli_commands_help(runner: Runner, app_cli_cmd):
    for name, _ in app_cli_cmd:
        runner.invoke(name + " --help", raise_on_error=True)


def test_version(runner: Runner):
    from mlem import __version__

    result = runner.invoke("--version")
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    assert __version__ in result.stdout
