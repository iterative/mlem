import traceback

from pydantic import ValidationError

from mlem.config import repo_config
from mlem.contrib.pandas import PandasConfig
from mlem.core.errors import MlemError
from tests.cli.conftest import Runner


def test_set_get(runner: Runner, mlem_repo):
    result = runner.invoke(
        f"config set pandas.default_format json --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception

    result = runner.invoke(
        f"config set core.additional_extensions ext1 --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception

    result = runner.invoke(
        f"config get pandas.default_format --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "json"

    result = runner.invoke(
        f"config get core.additional_extensions --repo {mlem_repo}".split()
    )

    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "ext1"

    assert repo_config(mlem_repo).additional_extensions == ["ext1"]
    assert (
        repo_config(mlem_repo, section=PandasConfig).default_format == "json"
    )


def test_set_get_validation(runner: Runner, mlem_repo):
    result = runner.invoke(
        f"config set core.nonexisting json --repo {mlem_repo}".split()
    )

    assert result.exit_code == 1
    assert isinstance(
        result.exception, ValidationError
    ), traceback.format_exception(
        type(result.exception),
        result.exception,
        result.exception.__traceback__,
    )

    result = runner.invoke(
        f"config set nonexisting json --repo {mlem_repo}".split()
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, MlemError)

    result = runner.invoke(
        f"config set core.nonexisting json --repo {mlem_repo} --no-validate".split()
    )

    assert result.exit_code == 0
