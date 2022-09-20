from io import StringIO
from unittest import mock

from mlem.core.errors import MlemError
from mlem.ui import echo, stderr_echo

EXCEPTION_MESSAGE = "Test Exception Message"
MLEM_ERROR_MESSAGE = "Test Mlem Error Message"
STDERR_MESSAGE = "Test Stderr Message"
TRACEBACK_IDENTIFIER_MESSAGE = "raise effect"


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


def test_stderr_echo():
    with mock.patch("sys.stderr", new_callable=StringIO) as mock_stderr:
        with stderr_echo():
            echo(STDERR_MESSAGE)
            mock_stderr.seek(0)
            output = mock_stderr.read()
            assert len(output) > 0, "Output is empty, but should not be"
            assert STDERR_MESSAGE in output


def test_traceback_option_exception(runner):
    # patch the ls command and ensure it throws an expection.
    with mock.patch(
        "mlem.api.commands.ls", side_effect=Exception(EXCEPTION_MESSAGE)
    ):
        result = runner.invoke(
            ["--tb", "list"],
        )
        assert result.exit_code == 1, (
            result.stdout,
            result.stderr,
            result.exception,
        )

        assert len(result.stderr) > 0, "Output is empty, but should not be"
        assert TRACEBACK_IDENTIFIER_MESSAGE in result.stderr
        assert EXCEPTION_MESSAGE in result.stderr


def test_traceback_option_mlem_error(runner):
    # patch the ls command and ensure it throws an expection.
    with mock.patch(
        "mlem.api.commands.ls", side_effect=MlemError(MLEM_ERROR_MESSAGE)
    ):
        result = runner.invoke(
            ["--tb", "list"],
        )
        assert result.exit_code == 1, (
            result.stdout,
            result.stderr,
            result.exception,
        )

        assert len(result.stderr) > 0, "Output is empty, but should not be"
        assert TRACEBACK_IDENTIFIER_MESSAGE in result.stderr
        assert MLEM_ERROR_MESSAGE in result.stderr


def test_verbose_option_exception(runner):
    # patch the ls command and ensure it throws an expection.
    with mock.patch(
        "mlem.api.commands.ls", side_effect=Exception(EXCEPTION_MESSAGE)
    ):
        result = runner.invoke(
            ["--verbose", "list"],
        )
        assert result.exit_code == 1, (
            result.stdout,
            result.stderr,
            result.exception,
        )

        assert len(result.stderr) > 0, "Output is empty, but should not be"
        assert TRACEBACK_IDENTIFIER_MESSAGE in result.stderr
        assert EXCEPTION_MESSAGE in result.stderr


def test_verbose_option_mlem_error(runner):
    # patch the ls command and ensure it throws an expection.
    with mock.patch(
        "mlem.api.commands.ls", side_effect=MlemError(MLEM_ERROR_MESSAGE)
    ):
        result = runner.invoke(
            ["--verbose", "list"],
        )
        assert result.exit_code == 1, (
            result.stdout,
            result.stderr,
            result.exception,
        )

        assert len(result.stderr) > 0, "Output is empty, but should not be"
        assert TRACEBACK_IDENTIFIER_MESSAGE in result.stderr
        assert MLEM_ERROR_MESSAGE in result.stderr
