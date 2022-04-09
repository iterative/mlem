from mlem.contrib.heroku.meta import HerokuEnvMeta
from mlem.core.metadata import load_meta
from mlem.utils.path import make_posix
from tests.cli.conftest import Runner


def test_create(runner: Runner, tmp_path):
    result = runner.invoke(
        f"create env heroku {make_posix(str(tmp_path))} -c api_key=aaa"
    )
    assert result.exit_code == 0, result.exception
    env = load_meta(str(tmp_path))
    assert isinstance(env, HerokuEnvMeta)
    assert env.api_key == "aaa"
