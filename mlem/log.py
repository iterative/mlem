import logging.config

LOG_LEVEL = logging.DEBUG
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

logger = logging.getLogger("mlem")
