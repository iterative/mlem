import os

from mlem.constants import MLEM_CONFIG_FILE_NAME
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


def test_init(runner: Runner, tmpdir):
    result = runner.invoke(f"init {make_posix(str(tmpdir))}")
    assert result.exit_code == 0, result.exception
    assert os.path.isfile(tmpdir / MLEM_CONFIG_FILE_NAME)
