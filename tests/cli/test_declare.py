from mlem.contrib.docker import DockerDirBuilder
from mlem.contrib.docker.context import DockerBuildArgs
from mlem.contrib.fastapi import FastAPIServer
from mlem.contrib.heroku.meta import HerokuEnv
from mlem.core.metadata import load_meta
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


def test_declare(runner: Runner, tmp_path):
    result = runner.invoke(
        f"declare env heroku {make_posix(str(tmp_path))} --api_key aaa"
    )
    assert result.exit_code == 0, result.exception
    env = load_meta(str(tmp_path))
    assert isinstance(env, HerokuEnv)
    assert env.api_key == "aaa"


def test_declare_list(runner: Runner, tmp_path):
    result = runner.invoke(
        f"declare builder docker_dir {make_posix(str(tmp_path))} --server fastapi --target lol --args.templates_dir.0 kek --args.templates_dir.1 kek2"
    )
    assert result.exit_code == 0, (result.exception, result.output)
    builder = load_meta(str(tmp_path))
    assert isinstance(builder, DockerDirBuilder)
    assert isinstance(builder.server, FastAPIServer)
    assert builder.target == "lol"
    assert isinstance(builder.args, DockerBuildArgs)
    assert builder.args.templates_dir == ["kek", "kek2"]
