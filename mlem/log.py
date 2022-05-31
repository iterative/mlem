"""
Loggers used in other parts of MLEM
"""
import logging.config

from mlem.config import LOCAL_CONFIG

LOG_LEVEL = LOCAL_CONFIG.LOG_LEVEL
if LOCAL_CONFIG.DEBUG:
    LOG_LEVEL = logging.getLevelName(logging.DEBUG)

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "mlem": {
            "handlers": ["default"],
            "level": LOG_LEVEL,
        }
    },
}

logging.config.dictConfig(logging_config)
