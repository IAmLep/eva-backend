import logging
import logging.config
from typing import Optional

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG",
    },
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "uvicorn.error": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "fastapi": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        }
    },
}

def setup_logging(config: Optional[dict] = None):
    """
    Configure logging using the provided configuration or the default.
    
    Args:
        config: A dictionary containing logging configuration.
               If None, the default configuration is used.
    """
    if config is None:
        config = LOGGING_CONFIG
    
    logging.config.dictConfig(config)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully")
    
    return logger