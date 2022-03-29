import pickle

import pytest
from typer.testing import CliRunner

from mlem.cli import app
from mlem.core.metadata import load


@pytest.fixture
def write_model_pickle(model):
    def write(path):
        with open(path, "wb") as f:
            pickle.dump(model, f)

    return write


@pytest.mark.parametrize("file_ext, type_", [(".pkl", None), ("", "pickle")])
def test_import_model_pickle_copy(
    write_model_pickle, train, tmpdir, file_ext, type_
):
    path = str(tmpdir / "mymodel" + file_ext)
    write_model_pickle(path)

    out_path = str(tmpdir / "mlem_model")

    runner = CliRunner()

    result = runner.invoke(
        app,
        ["import", path, out_path, "--type", type_, "--copy"],
    )
    assert result.exit_code == 0, (result.output, result.exception)

    loaded = load(out_path)
    loaded.predict(train)
