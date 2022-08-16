import os.path
from typing import ClassVar

from mlem.cli.build import create_build_command
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


class BuilderMock(MlemBuilder):
    type: ClassVar = "mock"
    target: str

    def build(self, obj: MlemModel):
        with open(self.target, "w", encoding="utf8") as f:
            f.write(obj.loc.path)


create_build_command(BuilderMock.type, BuilderMock)


def test_build(runner: Runner, model_meta_saved_single, tmp_path):
    path = os.path.join(tmp_path, "packed")
    result = runner.invoke(
        f"build mock {make_posix(model_meta_saved_single.loc.uri)} --target {make_posix(path)}"
    )

    assert result.exit_code == 0, (result.exception, result.output)

    with open(path, encoding="utf8") as f:
        assert f.read().strip() == model_meta_saved_single.loc.path
