from iterative_telemetry import IterativeTelemetryLogger

from mlem import CONFIG
from mlem.version import __version__

telemetry = IterativeTelemetryLogger(
    "mlem", __version__, not CONFIG.NO_ANALYTICS, debug=True
)
