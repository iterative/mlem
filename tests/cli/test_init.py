import os

from mlem.config import CONFIG_FILE_NAME
from mlem.constants import MLEM_DIR
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


def test_init(runner: Runner, tmpdir):
    result = runner.invoke(f"init {make_posix(str(tmpdir))}")
    assert result.exit_code == 0, result.exception
    assert os.path.isdir(tmpdir / MLEM_DIR)
    assert os.path.isfile(tmpdir / MLEM_DIR / CONFIG_FILE_NAME)
