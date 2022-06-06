import posixpath
import tempfile

from mlem.core.metadata import load_meta
from tests.cli.conftest import Runner


def test_model_cloning(runner: Runner, model_path):
    with tempfile.TemporaryDirectory() as path:
        path = posixpath.join(path, "cloned")
        result = runner.invoke(["clone", model_path, path, "--no-index"])
        assert result.exit_code == 0, (result.exception, result.output)
        load_meta(path, load_value=False)
