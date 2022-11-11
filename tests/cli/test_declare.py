from functools import lru_cache
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from mlem.cli.declare import create_declare_mlem_object_subcommand, declare
from mlem.contrib.docker import DockerDirBuilder
from mlem.contrib.docker.context import DockerBuildArgs
from mlem.contrib.fastapi import FastAPIServer
from mlem.contrib.heroku.meta import HerokuEnv
from mlem.contrib.pip.base import PipBuilder
from mlem.core.base import build_mlem_object
from mlem.core.metadata import load_meta
from mlem.core.objects import EnvLink, MlemBuilder, MlemModel
from mlem.runtime.server import Server
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner
from tests.cli.test_deployment import MlemDeploymentMock, MlemEnvMock

builder_typer = [
    g.typer_instance
    for g in declare.registered_groups
    if g.typer_instance.info.name == "builder"
][0]
builder_typer.pretty_exceptions_short = False

all_test_params = []


class SimpleValue(BaseModel):
    value: str


class ComplexValue(BaseModel):
    field: str
    field_list: List[str] = []
    field_dict: Dict[str, str] = {}


class ListValue(BaseModel):
    f: List[str] = []


class _MockBuilder(MlemBuilder):
    """mock"""

    def build(self, obj: MlemModel):
        pass

    def __init_subclass__(cls):
        cls.__doc__ = "mock"
        super().__init_subclass__()


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
        ("", []),
        (
            "--args.templates_dir.0 kek --args.templates_dir.1 kek2",
            ["kek", "kek2"],
        ),
    ],
)
def test_declare_list(runner: Runner, tmp_path, args, res):
    result = runner.invoke(
        f"declare builder docker_dir {make_posix(str(tmp_path))} --server fastapi --target lol "
        + args,
        raise_on_error=True,
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, DockerDirBuilder)
    assert isinstance(builder.server, FastAPIServer)
    assert builder.target == "lol"
    assert isinstance(builder.args, DockerBuildArgs)
    assert builder.args.templates_dir == res


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
        + args,
        raise_on_error=True,
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, PipBuilder)
    assert builder.package_name == "lol"
    assert builder.target == "lol"
    assert builder.additional_setup_kwargs == res


class MockListComplexValue(_MockBuilder):
    """mock"""

    field: List[ComplexValue] = []


all_test_params.append(
    pytest.param(
        MockListComplexValue(), "", id=f"{MockListComplexValue.type}_empty"
    )
)
all_test_params.append(
    pytest.param(
        MockListComplexValue(
            field=[
                ComplexValue(
                    field="a",
                    field_list=["a", "a"],
                    field_dict={"a": "a", "b": "b"},
                ),
                ComplexValue(
                    field="a",
                    field_list=["a", "a"],
                    field_dict={"a": "a", "b": "b"},
                ),
            ]
        ),
        "--field.0.field a --field.0.field_list.0 a --field.0.field_list.1 a --field.0.field_dict.a a --field.0.field_dict.b b "
        "--field.1.field a --field.1.field_list.0 a --field.1.field_list.1 a --field.1.field_dict.a a --field.1.field_dict.b b",
        id=f"{MockListComplexValue.type}_full",
    )
)


class MockListListValue(_MockBuilder):
    """mock"""

    f: List[ListValue] = []


all_test_params.append(
    pytest.param(MockListListValue(), "", id="list_list_value_empty")
)
all_test_params.append(
    pytest.param(
        MockListListValue(
            f=[ListValue(f=["a", "b"]), ListValue(f=["a", "b"])]
        ),
        "--f.0.f.0 a --f.0.f.1 b --f.1.f.0 a --f.1.f.1 b",
        id="list_list_value_full",
    )
)


class MockModelListBuilder(_MockBuilder):
    """mock"""

    field: List[SimpleValue] = []


all_test_params.append(
    pytest.param(MockModelListBuilder(), "", id="model_list_empty")
)
all_test_params.append(
    pytest.param(
        MockModelListBuilder(
            field=[SimpleValue(value="kek"), SimpleValue(value="kek2")]
        ),
        "--field.0.value kek --field.1.value kek2",
        id="model_list_full",
    )
)


class MockModelDictBuilder(_MockBuilder):
    """mock"""

    field: Dict[str, SimpleValue] = {}


all_test_params.append(
    pytest.param(MockModelDictBuilder(), "", id="model_dict_empty")
)
all_test_params.append(
    pytest.param(
        MockModelDictBuilder(
            field={
                "k1": SimpleValue(value="kek"),
                "k2": SimpleValue(value="kek2"),
            }
        ),
        "--field.k1.value kek --field.k2.value kek2",
        id="model_dict_empty",
    )
)


class MockFlatList(_MockBuilder):
    """mock"""

    f: List[List[str]] = []


all_test_params.append(
    pytest.param(MockFlatList(f=[]), "", id="flat_list_empty")
)
all_test_params.append(
    pytest.param(
        MockFlatList(f=[["a", "a"], ["a", "a"]]),
        "--f.0.0 a --f.0.1 a --f.1.0 a --f.1.1 a",
        id="flat_list_full",
    )
)


class MockFlatListDict(_MockBuilder):
    """mock"""

    f: List[Dict[str, str]] = []


all_test_params.append(
    pytest.param(MockFlatListDict(), "", id="flat_list_dict_empty")
)
all_test_params.append(
    pytest.param(
        MockFlatListDict(f=[{"k1": "a"}, {"k2": "b"}]),
        "--f.0.k1 a --f.1.k2 b",
        id="flat_list_dict_full",
    )
)


class MockFlatDictList(_MockBuilder):
    """mock"""

    f: Dict[str, List[str]] = {}


all_test_params.append(
    pytest.param(MockFlatDictList(), "", id="flat_dict_list_empty")
)
all_test_params.append(
    pytest.param(
        MockFlatDictList(f={"k1": ["a"], "k2": ["b"]}),
        "--f.k1.0 a --f.k2.0 b",
        id="flat_dict_list_full",
    )
)


class MockFlatDict(_MockBuilder):
    """mock"""

    f: Dict[str, Dict[str, str]] = {}


all_test_params.append(pytest.param(MockFlatDict(), "", id="flat_dict_empty"))
all_test_params.append(
    pytest.param(
        MockFlatDict(f={"k1": {"k1": "a"}, "k2": {"k2": "b"}}),
        "--f.k1.k1 a --f.k2.k2 b",
        id="flat_dict_full",
    )
)


class MaskedField(_MockBuilder):
    """mock"""

    field: ListValue
    project: str


all_test_params.append(
    pytest.param(
        MaskedField(project="a", field=ListValue(f=["a"])),
        "--.project a --field.f.0 a",
        id="masked",
    )
)


class BooleanField(_MockBuilder):
    field: bool


all_test_params.extend(
    (
        pytest.param(
            BooleanField(field=True),
            "--field 1",
            id="bool_true_1",
        ),
        pytest.param(
            BooleanField(field=False),
            "--field 0",
            id="bool_false_0",
        ),
        pytest.param(
            BooleanField(field=True),
            "--field True",
            id="bool_true",
        ),
        pytest.param(
            BooleanField(field=False),
            "--field False",
            id="bool_false",
        ),
    )
)


class AllowNoneField(_MockBuilder):
    field: Optional[int] = 0


all_test_params.extend(
    (
        pytest.param(
            AllowNoneField(field=10), "--field 10", id="allow_none_value"
        ),
        pytest.param(
            AllowNoneField(field=None), "--field None", id="allow_none_none"
        ),
        pytest.param(AllowNoneField(), "", id="allow_none_default"),
    )
)


class DictIntKey(_MockBuilder):
    field: Dict[int, int] = {}


all_test_params.extend(
    (
        pytest.param(
            DictIntKey(field={10: 10}), "--field.10 10", id="dict_int_key"
        ),
        pytest.param(DictIntKey(), "", id="dict_int_key_empty"),
    )
)


class NestedDictIntKey(_MockBuilder):
    field: DictIntKey


all_test_params.extend(
    (
        pytest.param(
            NestedDictIntKey(field=DictIntKey(field={10: 10})),
            "--field.field.10 10",
            id="nested_dict_int_key",
        ),
    )
)


class DoubleDictIntKey(_MockBuilder):
    field: Dict[int, Dict[int, int]]


# TODO: https://github.com/iterative/mlem/issues/483
# all_test_params.extend(
#     (
#         pytest.param(
#             DoubleDictIntKey(field={10:{10:10}}), "--field.10.10 10", id="double_dict_int_key"
#         ),
#     )
# )


class RootList(BaseModel):
    __root__: List[int]


class RootListNested(_MockBuilder):
    field: RootList


all_test_params.extend(
    (
        pytest.param(
            RootListNested(field=RootList(__root__=[10])),
            "--field.0 10",
            id="root_list_nested",
        ),
    )
)


@lru_cache()
def _declare_builder_command(type_: str):
    create_declare_mlem_object_subcommand(
        builder_typer,
        type_,
        MlemBuilder.object_type,
        MlemBuilder,
    )


@pytest.mark.parametrize("expected, args", all_test_params)
def test_declare_models(
    runner: Runner, tmp_path, args: str, expected: MlemBuilder
):
    _declare_builder_command(expected.__get_alias__())
    result = runner.invoke(
        f"declare builder {expected.__get_alias__()} {make_posix(str(tmp_path))} "
        + args,
        raise_on_error=True,
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, type(expected))
    assert builder == expected


class RootValue(BaseModel):
    __root__: List[str] = []


class MockComplexBuilder(_MockBuilder):
    """mock"""

    string: str
    str_list: List[str] = []
    str_dict: Dict[str, str] = {}
    str_list_dict: List[Dict[str, str]] = []
    str_dict_list: Dict[str, List[str]] = {}
    value: ComplexValue

    value_list: List[ComplexValue] = []
    value_dict: Dict[str, ComplexValue] = {}
    root_value: RootValue
    root_list: List[RootValue] = []
    root_dict: Dict[str, RootValue] = {}
    server: Server
    server_list: List[Server] = []
    server_dict: Dict[str, Server] = {}


create_declare_mlem_object_subcommand(
    builder_typer,
    MockComplexBuilder.type,
    MlemBuilder.object_type,
    MlemBuilder,
)


def test_declare_all_together(runner: Runner, tmp_path):
    args = [
        "string",
        "str_list.0",
        "str_list.1",
        "str_dict.k1",
        "str_dict.k2",
        "str_list_dict.0.k1",
        "str_list_dict.0.k2",
        "str_list_dict.1.k1",
        "str_list_dict.1.k2",
        "str_dict_list.k1.0",
        "str_dict_list.k1.1",
        "str_dict_list.k2.0",
        "str_dict_list.k2.1",
        "value.field",
        "value.field_list.0",
        "value.field_list.1",
        "value.field_dict.k1",
        "value.field_dict.k2",
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
        "value_dict.k1.field",
        "value_dict.k1.field_list.0",
        "value_dict.k1.field_list.1",
        "value_dict.k1.field_dict.k1",
        "value_dict.k1.field_dict.k2",
        "value_dict.k2.field",
        "value_dict.k2.field_list.0",
        "value_dict.k2.field_list.1",
        "value_dict.k2.field_dict.k1",
        "value_dict.k2.field_dict.k2",
        "root_value.0",
        "root_value.1",
        "root_list.0.0",
        "root_list.0.1",
        "root_list.1.0",
        "root_list.1.1",
        "root_dict.k1.0",
        "root_dict.k1.1",
        "root_dict.k2.0",
        "root_dict.k2.1",
    ]
    server_args: Dict[str, Any] = {
        "server": "fastapi",
        "server.port": 0,
        "server_list.0": "fastapi",
        "server_list.0.port": 0,
        "server_list.1": "fastapi",
        "server_list.1.port": 0,
        "server_dict.k1": "fastapi",
        "server_dict.k1.port": 0,
        "server_dict.k2": "fastapi",
        "server_dict.k2.port": 0,
    }
    args_str = " ".join(f"--{k} lol" for k in args)
    args_str += " " + " ".join(f"--{k} {v}" for k, v in server_args.items())
    result = runner.invoke(
        f"declare builder {MockComplexBuilder.type} {make_posix(str(tmp_path))} {args_str}",
        raise_on_error=True,
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, MockComplexBuilder)
    assert builder == build_mlem_object(
        MlemBuilder,
        MockComplexBuilder.type,
        str_conf=[f"{k}=lol" for k in args],
        conf=server_args,
    )


@pytest.mark.parametrize(
    "args,env_value",
    [
        ("", None),
        ("--env path", "path"),
        (
            "--env.path path --env.project project",
            EnvLink(path="path", project="project"),
        ),
        ("--env.env_param val", MlemEnvMock(env_param="val")),
    ],
)
def test_declare_deployment_env(
    runner: Runner, tmp_path, args: str, env_value
):
    path = make_posix(str(tmp_path))
    runner.invoke(
        f"declare deployment {MlemDeploymentMock.type} {path} " + args,
        raise_on_error=True,
    )
    meta = load_meta(path, force_type=MlemDeploymentMock)

    assert meta.env == env_value


def test_declare_unknown_option_raises(runner: Runner):
    with pytest.raises(RuntimeError, match=".*No such option.*"):
        runner.invoke(
            f"--tb declare deployment {MlemDeploymentMock.type} nowhere --nonexistent_option value",
            raise_on_error=True,
        )
