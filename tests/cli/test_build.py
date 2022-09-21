import json
import os.path
from typing import ClassVar

from pydantic import parse_obj_as
from yaml import safe_dump

from mlem.cli.build import create_build_command
from mlem.contrib.fastapi import FastAPIServer
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.runtime.server import Server
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


class BuilderMock(MlemBuilder):
    """mock"""

    type: ClassVar = "mock"
    target: str
    """target"""
    server: Server
    """server"""

    def build(self, obj: MlemModel):
        with open(self.target, "w", encoding="utf8") as f:
            f.write(obj.loc.path + "\n")
            json.dump(self.server.dict(), f)


create_build_command(BuilderMock.type)


def test_build(runner: Runner, model_meta_saved_single, tmp_path):
    path = os.path.join(tmp_path, "packed")
    result = runner.invoke(
        f"build mock -m {make_posix(model_meta_saved_single.loc.uri)} --target {make_posix(path)} --server fastapi --server.port 1000"
    )

    assert result.exit_code == 0, (
        result.stdout,
        result.stderr,
        result.exception,
    )

    with open(path, encoding="utf8") as f:
        lines = f.read().splitlines()
        assert len(lines) == 2
        path, serv = lines
        assert path == model_meta_saved_single.loc.path
        assert parse_obj_as(Server, json.loads(serv)) == FastAPIServer(
            port=1000
        )


def test_build_with_file_conf(
    runner: Runner, model_meta_saved_single, tmp_path
):
    path = os.path.join(tmp_path, "packed")
    server_path = os.path.join(tmp_path, "server.yaml")
    with open(server_path, "w", encoding="utf8") as f:
        safe_dump(FastAPIServer(port=9999).dict(), f)

    result = runner.invoke(
        f"build mock -m {make_posix(model_meta_saved_single.loc.uri)} --target {make_posix(path)} --file_conf server={make_posix(server_path)}"
    )

    assert result.exit_code == 0, (result.exception, result.output)

    with open(path, encoding="utf8") as f:
        lines = f.read().splitlines()
        assert len(lines) == 2
        path, serv = lines
        assert path == model_meta_saved_single.loc.path
        assert parse_obj_as(Server, json.loads(serv)) == FastAPIServer(
            port=9999
        )


def test_build_with_load(runner: Runner, model_meta_saved_single, tmp_path):
    path = os.path.join(tmp_path, "packed")
    load_path = os.path.join(tmp_path, "builder.yaml")
    builder = BuilderMock(
        server=FastAPIServer(port=9999), target=make_posix(path)
    )
    with open(load_path, "w", encoding="utf8") as f:
        safe_dump(builder.dict(), f)

    result = runner.invoke(
        f"build -m {make_posix(model_meta_saved_single.loc.uri)} --load {make_posix(load_path)}"
    )

    assert result.exit_code == 0, (result.exception, result.output)

    with open(path, encoding="utf8") as f:
        lines = f.read().splitlines()
        assert len(lines) == 2
        path, serv = lines
        assert path == model_meta_saved_single.loc.path
        assert parse_obj_as(Server, json.loads(serv)) == FastAPIServer(
            port=9999
        )
