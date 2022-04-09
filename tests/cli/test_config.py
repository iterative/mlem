from mlem.config import repo_config
from mlem.contrib.pandas import PandasConfig
from tests.cli.conftest import Runner


def test_set_get(runner: Runner, mlem_repo):
    result = runner.invoke(
        f"config set pandas.default_format json --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception

    result = runner.invoke(
        f"config set additional_extensions_raw ext1 --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception

    result = runner.invoke(
        f"config get pandas.default_format --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "json"

    result = runner.invoke(
        f"config get additional_extensions_raw --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "ext1"

    assert repo_config(mlem_repo).ADDITIONAL_EXTENSIONS == ["ext1"]
    assert PandasConfig(config_path=mlem_repo).DEFAULT_FORMAT == "json"
