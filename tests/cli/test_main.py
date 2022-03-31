import pytest
from typer.main import get_command_from_info

from mlem.cli import app


@pytest.fixture
def app_cmd():
    return app.registered_commands


@pytest.fixture
def app_cli_cmd(app_cmd):
    return (get_command_from_info(c) for c in app_cmd)


def test_commands_help(app_cli_cmd):
    no_help = []
    for cli_cmd in app_cli_cmd:
        if cli_cmd.help is None:
            no_help.append(cli_cmd.name)
    assert len(no_help) == 0, f"{no_help} cli commnads do not have help!"


def test_commands_args_help(app_cli_cmd):
    no_help = []
    for cmd in app_cli_cmd:
        for arg in cmd.params:
            if arg.help is None:
                no_help.append(f"{cmd.name}:{arg.name}")
    assert len(no_help) == 0, f"{no_help} cli commnad args do not have help!"
