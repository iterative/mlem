from iterative_telemetry import IterativeTelemetryLogger

from mlem import LOCAL_CONFIG
from mlem.version import __version__

telemetry = IterativeTelemetryLogger(
    "mlem",
    __version__,
    not LOCAL_CONFIG.TESTS and not LOCAL_CONFIG.NO_ANALYTICS,
    url="https://telemetry.mlem.dev/api/v1/s2s/event?ip_policy=strict",
)
