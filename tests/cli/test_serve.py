from typing import ClassVar

from mlem.runtime import Interface
from mlem.runtime.server.base import Server
from mlem.ui import echo
from tests.cli.conftest import Runner


class MockServer(Server):
    type: ClassVar = "mock"
    param: str = "wrong"

    def serve(self, interface: Interface):
        echo(self.param)


def test_serve(runner: Runner, model_single_path):
    result = runner.invoke(f"serve {model_single_path} mock -c param=aaa")
    assert result.exit_code == 0, result.exception
    assert result.output.splitlines()[-1] == "aaa"
