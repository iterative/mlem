import pytest
from click.testing import Result
from typer.testing import CliRunner

from mlem import LOCAL_CONFIG
from mlem.cli import app

app.pretty_exceptions_short = False


class Runner:
    def __init__(self):
        self._runner = CliRunner(mix_stderr=False)

    def invoke(self, *args, raise_on_error: bool = False, **kwargs) -> Result:
        result = self._runner.invoke(app, *args, **kwargs)
        if raise_on_error and result.exit_code != 0:
            if result.exit_code == 1:
                raise result.exception
            raise RuntimeError(result.stderr)
        return result


@pytest.fixture
def runner() -> Runner:
    return Runner()


@pytest.fixture
def no_debug():
    tmp = LOCAL_CONFIG.DEBUG
    try:
        LOCAL_CONFIG.DEBUG = False
        yield
    finally:
        LOCAL_CONFIG.DEBUG = tmp
