import contextlib
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
_pass_params = False


@contextlib.contextmanager
def pass_telemetry_params():
    global _pass_params  # pylint: disable=global-statement
    pass_params = _pass_params
    try:
        _pass_params = True
        yield
    finally:
        _pass_params = pass_params


def api_telemetry(f):
    @wraps(f)
    def inner(*args, **kwargs):
        global _is_api_running, _pass_params  # pylint: disable=global-statement
        is_nested = _is_api_running
        pass_params = _pass_params
        _pass_params = False
        _is_api_running = True
        try:
            from mlem.cli.utils import is_cli

            with telemetry.event_scope("api", f.__name__) as event:
                try:
                    return f(*args, **kwargs)
                except Exception as exc:
                    event.error = exc.__class__.__name__
                    raise
                finally:
                    if not is_nested and not is_cli():
                        telemetry.send_event(
                            event.interface,
                            event.action,
                            event.error,
                            **event.kwargs,
                        )

        finally:
            if pass_params:
                for key, value in event.kwargs.items():
                    telemetry.log_param(key, value)
            _is_api_running = is_nested
            _pass_params = pass_params

    return inner
