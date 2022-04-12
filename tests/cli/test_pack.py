import os.path
from typing import ClassVar

from mlem.core.objects import ModelMeta
from mlem.pack import Packager
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


class PackagerMock(Packager):
    type: ClassVar = "mock"
    target: str

    def package(self, obj: ModelMeta):
        with open(self.target, "w", encoding="utf8") as f:
            f.write(obj.loc.path)


def test_pack(runner: Runner, model_meta_saved_single, tmp_path):
    path = os.path.join(tmp_path, "packed")
    result = runner.invoke(
        f"pack {make_posix(model_meta_saved_single.loc.uri)} -c target={make_posix(path)} mock"
    )

    assert result.exit_code == 0, (result.exception, result.output)

    with open(path, encoding="utf8") as f:
        assert f.read().strip() == model_meta_saved_single.loc.path
