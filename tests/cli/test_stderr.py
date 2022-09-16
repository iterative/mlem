from unittest import mock

from mlem.core.errors import MlemError

EXCEPTION_MESSAGE = "Test Exception Message"


def test_stderr_exception(runner):
    # patch the ls command and ensure it throws an expection.
    with mock.patch(
        "mlem.api.commands.ls", side_effect=Exception(EXCEPTION_MESSAGE)
    ):
        result = runner.invoke(
            ["list"],
        )
        assert result.exit_code == 1, (
            result.stdout,
            result.stderr,
            result.exception,
        )
        assert len(result.stderr) > 0, "Output is empty, but should not be"
        assert EXCEPTION_MESSAGE in result.stderr


MLEM_ERROR_MESSAGE = "Test Mlem Error Message"


def test_stderr_mlem_error(runner):
    # patch the ls command and ensure it throws a mlem error.
    with mock.patch(
        "mlem.api.commands.ls", side_effect=MlemError(MLEM_ERROR_MESSAGE)
    ):
        result = runner.invoke(
            ["list"],
        )
        assert result.exit_code == 1, (
            result.stdout,
            result.stderr,
            result.exception,
        )
        assert len(result.stderr) > 0, "Output is empty, but should not be"
        assert MLEM_ERROR_MESSAGE in result.stderr
