import pytest
from click import Group
from typer.main import get_command_from_info, get_group_from_info

from mlem.cli import app
from tests.cli.conftest import Runner


def iter_group(group: Group, prefix=()):
    prefix = prefix + (group.name,)
    yield prefix, group
    for name, c in group.commands.items():
        if isinstance(c, Group):
            yield from iter_group(c, prefix + (name,))
        else:
            yield prefix + (name,), c


@pytest.fixture
def app_cli_cmd():
    commands = [get_command_from_info(c) for c in app.registered_commands]
    groups = [get_group_from_info(g) for g in app.registered_groups]
    return [(c.name, c) for c in commands] + [
        (" ".join(names), cmd) for g in groups for names, cmd in iter_group(g)
    ]


def test_commands_help(app_cli_cmd):
    no_help = []
    for name, cli_cmd in app_cli_cmd:
        if cli_cmd.help is None:
            no_help.append(name)
    assert len(no_help) == 0, f"{no_help} cli commnads do not have help!"


def test_commands_args_help(app_cli_cmd):
    no_help = []
    for name, cmd in app_cli_cmd:
        for arg in cmd.params:
            if arg.help is None:
                no_help.append(f"{name}:{arg.name}")
    assert len(no_help) == 0, f"{no_help} cli commnad args do not have help!"


def test_commands_examples(app_cli_cmd):
    no_examples = []
    for name, cmd in app_cli_cmd:
        if cmd.examples is None and not isinstance(cmd, Group):
            no_examples.append(name)
    assert (
        len(no_examples) == 0
    ), f"{no_examples} cli commnads do not have examples!"


@pytest.mark.parametrize("cmd", ["--help", "-h"])
def test_help(runner: Runner, cmd):
    result = runner.invoke(cmd)
    assert result.exit_code == 0, (result.exception, result.output)


def test_cli_commands_help(runner: Runner, app_cli_cmd):
    for name, _ in app_cli_cmd:
        result = runner.invoke(name + " --help")
        assert result.exit_code == 0, (name, result.exception, result.output)


def test_version(runner: Runner):
    from mlem import __version__

    result = runner.invoke("--version")
    assert result.exit_code == 0, (result.exception, result.output)
    assert __version__ in result.output
