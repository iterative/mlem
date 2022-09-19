from typing import ClassVar

from mlem.cli.serve import create_serve_command
from mlem.runtime import Interface
from mlem.runtime.server import Server
from mlem.ui import echo
from tests.cli.conftest import Runner


class MockServer(Server):
    """mock"""

    type: ClassVar = "mock"
    param: str = "wrong"
    """param"""

    def serve(self, interface: Interface):
        echo(self.param)


create_serve_command(MockServer.type)


def test_serve(runner: Runner, model_single_path):
    result = runner.invoke(f"serve mock -m {model_single_path} --param aaa")
    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )
    assert result.stdout.splitlines()[-1] == "aaa"
