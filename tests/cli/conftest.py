import pytest
from click.testing import Result
from typer.testing import CliRunner

from mlem.cli import app


class Runner:
    def __init__(self):
        self._runner = CliRunner()

    def invoke(self, *args, **kwargs) -> Result:
        return self._runner.invoke(app, *args, **kwargs)


@pytest.fixture
def runner() -> Runner:
    return Runner()
