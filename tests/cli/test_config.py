import traceback

import pytest
from pydantic import ValidationError

from mlem.config import project_config
from mlem.contrib.pandas import PandasConfig
from mlem.core.errors import MlemError, UnknownConfigSection
from tests.cli.conftest import Runner


def test_set_get(runner: Runner, mlem_project):
    result = runner.invoke(
        f"config set pandas.default_format json --project {mlem_project}".split()
    )

    assert result.exit_code == 0, result.exception

    result = runner.invoke(
        f"config set core.additional_extensions ext1 --project {mlem_project}".split()
    )

    assert result.exit_code == 0, result.exception

    result = runner.invoke(
        f"config get pandas.default_format --project {mlem_project}".split()
    )

    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "json"

    result = runner.invoke(
        f"config get core.additional_extensions --project {mlem_project}".split()
    )

    assert result.exit_code == 0, result.exception
    assert result.stdout.strip() == "ext1"

    assert project_config(mlem_project).additional_extensions == ["ext1"]
    assert (
        project_config(mlem_project, section=PandasConfig).default_format
        == "json"
    )


def test_set_get_validation(runner: Runner, mlem_project):
    result = runner.invoke(
        f"--tb config set core.nonexisting json --project {mlem_project}".split()
    )

    assert result.exit_code == 1
    assert (
        isinstance(result.exception, ValidationError)
        or "extra fields not permitted" in result.stderr
    ), traceback.format_exception(
        type(result.exception),
        result.exception,
        result.exception.__traceback__,
    )

    result = runner.invoke(
        f"--tb config set nonexisting json --project {mlem_project}".split()
    )

    assert result.exit_code == 1
    assert (
        isinstance(result.exception, MlemError)
        or "[name] should contain at least one dot" in result.stderr
    ), traceback.format_exception(
        type(result.exception),
        result.exception,
        result.exception.__traceback__,
    )

    result = runner.invoke(
        f"--tb config set core.nonexisting json --project {mlem_project} --no-validate".split()
    )

    assert result.exit_code == 0


def test_double_set(runner: Runner, mlem_project):
    result = runner.invoke(
        f"config set pandas.default_format json --project {mlem_project}".split()
    )

    assert result.exit_code == 0, result.exception
    assert (
        project_config(mlem_project, section=PandasConfig).default_format
        == "json"
    )

    runner.invoke(
        f"config set pandas.default_format json --project {mlem_project}".split(),
        raise_on_error=True,
    )


def test_get_on_empty(runner: Runner, mlem_project):
    result = runner.invoke(
        f"config get core.debug --project {mlem_project}".split(),
        raise_on_error=True,
    )
    assert result.exit_code == 0
    assert result.output.strip() == "True"

    result = runner.invoke(
        f"config get pandas.default_format --project {mlem_project}".split(),
        raise_on_error=True,
    )
    assert result.exit_code == 0
    assert result.output.strip() == "csv"

    with pytest.raises(UnknownConfigSection):
        runner.invoke(
            f"config get non_existent.kek --project {mlem_project}".split(),
            raise_on_error=True,
        )
