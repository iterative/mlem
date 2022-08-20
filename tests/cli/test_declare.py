from typing import ClassVar, Dict, List

import pytest
from pydantic import BaseModel

from mlem.cli.declare import create_declare_subcommand, declare
from mlem.contrib.docker import DockerDirBuilder
from mlem.contrib.docker.context import DockerBuildArgs
from mlem.contrib.fastapi import FastAPIServer
from mlem.contrib.heroku.meta import HerokuEnv
from mlem.contrib.pip.base import PipBuilder
from mlem.core.metadata import load_meta
from mlem.core.objects import MlemBuilder, MlemModel
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner

builder_typer = [
    g.typer_instance
    for g in declare.registered_groups
    if g.typer_instance.info.name == "builder"
][0]
builder_typer.pretty_exceptions_short = False


def test_declare(runner: Runner, tmp_path):
    result = runner.invoke(
        f"declare env heroku {make_posix(str(tmp_path))} --api_key aaa"
    )
    assert result.exit_code == 0, result.exception
    env = load_meta(str(tmp_path))
    assert isinstance(env, HerokuEnv)
    assert env.api_key == "aaa"


@pytest.mark.parametrize(
    "args, res",
    [
        # ("", []),
        (
            "--args.templates_dir.0 kek --args.templates_dir.1 kek2",
            ["kek", "kek2"],
        ),
    ],
)
def test_declare_list(runner: Runner, tmp_path, args, res):
    result = runner.invoke(
        f"declare builder docker_dir {make_posix(str(tmp_path))} --server fastapi --target lol "
        + args
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, DockerDirBuilder)
    assert isinstance(builder.server, FastAPIServer)
    assert builder.target == "lol"
    assert isinstance(builder.args, DockerBuildArgs)
    assert builder.args.templates_dir == res


class SimpleModel(BaseModel):
    value: str


class MockModelListBuilder(MlemBuilder):
    type: ClassVar = "mock_model_list_builder"
    field: List[SimpleModel] = []

    def build(self, obj: MlemModel):
        pass


create_declare_subcommand(
    builder_typer,
    MockModelListBuilder.type,
    MlemBuilder.object_type,
    MlemBuilder,
)


@pytest.mark.parametrize(
    "args, res",
    [
        ("", []),
        (
            "--field.0.value kek --field.1.value kek2",
            [SimpleModel(value="kek"), SimpleModel(value="kek2")],
        ),
    ],
)
def test_declare_list_model(runner: Runner, tmp_path, args, res):
    result = runner.invoke(
        f"declare builder {MockModelListBuilder.type} {make_posix(str(tmp_path))} "
        + args
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, MockModelListBuilder)
    assert builder.field == res


@pytest.mark.parametrize(
    "args, res",
    [
        ("", {}),
        (
            "--additional_setup_kwargs.key value --additional_setup_kwargs.key2 value2",
            {"key": "value", "key2": "value2"},
        ),
    ],
)
def test_declare_dict(runner: Runner, tmp_path, args, res):
    result = runner.invoke(
        f"declare builder pip {make_posix(str(tmp_path))} --package_name lol --target lol "
        + args
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, PipBuilder)
    assert builder.package_name == "lol"
    assert builder.target == "lol"
    assert builder.additional_setup_kwargs == res


class Value(BaseModel):
    field: str
    field_list: List[str] = []
    field_dict: Dict[str, str] = {}


class RootValue(BaseModel):
    __root__: List[str] = []


class MockComplexBuilder(MlemBuilder):
    type: ClassVar = "mock_complex"

    string: str
    # str_list: List[str] = []
    # str_dict: Dict[str, str] = {}
    # str_list_dict: List[Dict[str, str]] = []
    # str_dict_list: Dict[str, List[str]] = {}
    # value: Value
    value_list: List[Value] = []
    # value_dict: Dict[str, Value] = {}
    # root_value: RootValue
    # root_list: List[RootValue] = []
    # root_dict: Dict[str, RootValue] = {}
    # server: Server
    # server_list: List[Server] = []
    # server_dict: Dict[str, Server] = {}

    def build(self, obj: MlemModel):
        pass


create_declare_subcommand(
    builder_typer,
    MockComplexBuilder.type,
    MlemBuilder.object_type,
    MlemBuilder,
)


def test_declare_all_together(runner: Runner, tmp_path):
    args = [
        "string",
        # "str_list.0",
        # "str_list.1",
        # "str_dict.k1",
        # "str_dict.k2",
        # "str_list_dict.0.k1",
        # "str_list_dict.0.k2",
        # "str_list_dict.1.k1",
        # "str_list_dict.1.k2",
        # "str_dict_list.k1.0",
        # "str_dict_list.k1.1",
        # "str_dict_list.k2.0",
        # "str_dict_list.k2.1",
        # "value.field",
        # "value.field_list.0",
        # "value.field_list.1",
        # "value.field_dict.k1",
        # "value.field_dict.k2",
        "value_list.0.field",
        "value_list.0.field_list.0",
        "value_list.0.field_list.1",
        "value_list.0.field_dict.k1",
        "value_list.0.field_dict.k2",
        "value_list.1.field",
        "value_list.1.field_list.0",
        "value_list.1.field_list.1",
        "value_list.1.field_dict.k1",
        "value_list.1.field_dict.k2",
        # "value_dict.k1.field",
        # "value_dict.k1.field_list.0",
        # "value_dict.k1.field_list.1",
        # "value_dict.k1.field_dict.k1",
        # "value_dict.k1.field_dict.k2",
        # "value_dict.k2.field",
        # "value_dict.k2.field_list.0",
        # "value_dict.k2.field_list.1",
        # "value_dict.k2.field_dict.k1",
        # "value_dict.k2.field_dict.k2",
        # "root_value.0",
        # "root_value.1",
        # "root_list.0.0",
        # "root_list.0.1",
        # "root_list.1.0",
        # "root_list.1.1",
        # "root_dict.k1.0",
        # "root_dict.k1.1"
        # "root_dict.k2.0",
        # "root_dict.k2.1"
    ]
    server_args: Dict[str, str] = {
        # "server": "fastapi",
        # "server.port": 0,
        # "server_list.0": "fastapi",
        # "server_list.0.port": 0,
        # "server_list.1": "fastapi",
        # "server_list.1.port": 0,
        # "server_dict.k1": "fastapi",
        # "server_dict.k1.port": 0,
        # "server_dict.k2": "fastapi",
        # "server_dict.k2.port": 0,
    }
    args_str = " ".join(f"--{k} lol" for k in args)
    args_str += " " + " ".join(f"--{k} {v}" for k, v in server_args.items())
    result = runner.invoke(
        f"declare builder {MockComplexBuilder.type} {make_posix(str(tmp_path))} {args_str}"
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, MockComplexBuilder)
    # assert builder.package_name == "lol"
    # assert builder.target == "lol"
    # assert builder.additional_setup_kwargs == res
