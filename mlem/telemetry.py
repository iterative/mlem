import contextlib
import dataclasses
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, Union

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


@dataclasses.dataclass
class TelemetryEvent:
    interface: str
    action: str
    error: Optional[str] = None
    kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)


_current_event: Optional[TelemetryEvent] = None


def log_telemetry_param(key: str, value):
    if _current_event:
        _current_event.kwargs[key] = value


@contextlib.contextmanager
def telemetry_event_scope(
    interface: str, action: str
) -> Iterator[TelemetryEvent]:
    global _current_event  # pylint: disable=global-statement
    event = TelemetryEvent(interface=interface, action=action)
    tmp = _current_event
    _current_event = event
    try:
        yield event
    finally:
        _current_event = tmp


def log_telemetry(
    interface: str,
    action: str = None,
    skip: Union[bool, Callable[[TelemetryEvent], bool]] = None,
):
    def decorator(f):
        @wraps(f)
        def inner(*args, **kwargs):
            with telemetry_event_scope(
                interface, action or f.__name__
            ) as event:
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    event.error = e.__class__.__name__
                    raise
                finally:
                    if (
                        skip is None
                        or (callable(skip) and not skip(event))
                        or not skip
                    ):
                        telemetry.send_event(
                            event.interface,
                            event.action,
                            event.error,
                            **event.kwargs
                        )

        return inner

    return decorator


def skip_api_log(_):
    from mlem.cli.utils import is_cli

    return is_cli()


api_telemetry = log_telemetry("api", skip=skip_api_log)
