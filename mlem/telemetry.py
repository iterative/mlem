from functools import wraps

from iterative_telemetry import IterativeTelemetryLogger

from mlem.version import __version__


def _enabled():
    from mlem import LOCAL_CONFIG

    return not LOCAL_CONFIG.TESTS and not LOCAL_CONFIG.NO_ANALYTICS


telemetry = IterativeTelemetryLogger(
    "mlem",
    __version__,
    _enabled,
    url="https://telemetry.mlem.dev/api/v1/s2s/event?ip_policy=strict",
)


_is_api_running = False


def api_telemetry(f):
    @wraps(f)
    def inner(*args, **kwargs):
        global _is_api_running  # pylint: disable=global-statement
        is_nested = _is_api_running
        _is_api_running = True
        try:
            from mlem.cli.utils import is_cli

            return telemetry.log("api", skip=is_nested or is_cli())(f)(
                *args, **kwargs
            )
        finally:
            _is_api_running = is_nested

    return inner
