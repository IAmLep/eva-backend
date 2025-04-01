"""
Logging Configuration module for EVA backend.

This module configures the application's logging system with appropriate
formatters, handlers, and filters for different environments.

Last updated: 2025-04-01 10:44:46
Version: v1.8.6
Created by: IAmLep
"""

import json
import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from config import get_settings

# Define logger instance
logger = logging.getLogger(__name__)


class CloudRunFormatter(logging.Formatter):
    """
    Custom formatter for Google Cloud Run environments.
    
    Formats logs as structured JSON suitable for Cloud Logging.
    Includes additional metadata needed for proper log parsing.
    """
    
    def __init__(self):
        """Initialize the Cloud Run formatter."""
        super().__init__()
        # Get application settings
        self.settings = get_settings()
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON for Cloud Logging.
        
        Args:
            record: Log record to format
            
        Returns:
            str: JSON-formatted log entry
        """
        # Basic log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "app": self.settings.APP_NAME,
            "version": self.settings.APP_VERSION,
            "environment": self.settings.ENVIRONMENT,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from record
        if hasattr(record, 'props'):
            log_entry.update(record.props)
        
        return json.dumps(log_entry)


class RequestIdFilter(logging.Filter):
    """
    Filter that adds request ID to log records.
    
    Enriches log entries with request ID from context when available.
    """
    
    def __init__(self, name: str = "", request_id: Optional[str] = None):
        """
        Initialize the filter.
        
        Args:
            name: Name of the filter
            request_id: Optional default request ID
        """
        super().__init__(name)
        self.request_id = request_id or "no-request-id"
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log records and add request ID.
        
        Args:
            record: Log record to filter
            
        Returns:
            bool: Always True (doesn't filter out any records)
        """
        # Initialize props if not exists
        if not hasattr(record, 'props'):
            record.props = {}
        
        # Try to get request ID from context
        from contextvars import ContextVar
        try:
            request_id_var = ContextVar('request_id', default=self.request_id)
            record.props['request_id'] = request_id_var.get()
        except Exception:
            record.props['request_id'] = self.request_id
        
        return True


def configure_logging() -> None:
    """
    Configure the application's logging system.
    
    Sets up appropriate handlers, formatters, and log levels based on
    the current environment configuration.
    """
    settings = get_settings()
    
    # Determine log level from settings
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Base configuration with root logger
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": CloudRunFormatter,
            },
        },
        "filters": {
            "request_id": {
                "()": RequestIdFilter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard" if not settings.is_production else "json",
                "filters": ["request_id"],
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
            # Configure specific loggers with different levels
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "firestore_manager": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "api_tools": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            "cache_manager": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }
    
    # Add file handler for non-production environments if LOGS_DIR is set
    logs_dir = os.environ.get("LOGS_DIR")
    if logs_dir and not settings.is_production:
        logs_path = Path(logs_dir)
        logs_path.mkdir(exist_ok=True, parents=True)
        
        log_file = logs_path / f"{settings.APP_NAME.lower().replace(' ', '_')}.log"
        
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "standard",
            "filters": ["request_id"],
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
            "encoding": "utf8",
        }
        
        # Add file handler to all loggers
        for logger_name in config["loggers"]:
            config["loggers"][logger_name]["handlers"].append("file")
    
    # Configure logging with our settings
    logging.config.dictConfig(config)
    
    # Log startup message
    logger.info(
        f"Logging configured for {settings.APP_NAME} v{settings.APP_VERSION} "
        f"in {settings.ENVIRONMENT} environment with level {log_level_name}"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with the given name.
    
    Args:
        name: Logger name, typically __name__ of the calling module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds context to log messages.
    
    Useful for adding request-specific or user-specific information
    to all log messages from a particular context.
    """
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any] = None):
        """
        Initialize the logger adapter.
        
        Args:
            logger: Base logger to adapt
            extra: Additional context to add to all log messages
        """
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message by adding context.
        
        Args:
            msg: Log message
            kwargs: Additional arguments
            
        Returns:
            tuple: Processed message and kwargs
        """
        # Ensure props exists in kwargs
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        if 'props' not in kwargs['extra']:
            kwargs['extra']['props'] = {}
        
        # Add our extras to props
        kwargs['extra']['props'].update(self.extra)
        
        return msg, kwargs


def get_request_logger(request_id: str, user_id: Optional[str] = None) -> LoggerAdapter:
    """
    Get a logger configured with request and user context.
    
    Args:
        request_id: Request identifier
        user_id: Optional user identifier
        
    Returns:
        LoggerAdapter: Logger adapter with request context
    """
    # Create context dictionary
    context = {
        "request_id": request_id,
    }
    
    if user_id:
        context["user_id"] = user_id
    
    # Create and return adapter
    logger = get_logger("request")
    return LoggerAdapter(logger, context)


def set_log_level(level: Union[int, str]) -> None:
    """
    Dynamically change the log level at runtime.
    
    Args:
        level: Log level name (e.g., 'INFO') or value (e.g., 20)
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Set level on root logger and all handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers:
        handler.setLevel(level)
    
    logger.info(f"Log level changed to {logging.getLevelName(level)}")


def log_function_call(func):
    """
    Decorator to log function calls with parameters and results.
    
    Args:
        func: Function to decorate
        
    Returns:
        callable: Decorated function
    """
    func_logger = get_logger(func.__module__)
    
    async def async_wrapper(*args, **kwargs):
        func_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = await func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            func_logger.exception(f"{func.__name__} raised exception: {e}")
            raise
    
    def sync_wrapper(*args, **kwargs):
        func_logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            func_logger.debug(f"{func.__name__} returned: {result}")
            return result
        except Exception as e:
            func_logger.exception(f"{func.__name__} raised exception: {e}")
            raise
    
    # Check if the function is async or not
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper