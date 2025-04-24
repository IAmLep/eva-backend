"""
Logging configuration for the EVA backend application.

Sets up logging format (plain text or JSON) and level based on settings.
"""

import logging
import sys
import os
from pythonjsonlogger import jsonlogger # pip install python-json-logger

# --- Local Imports ---
from config import settings

# --- Configuration ---
def configure_logging():
    """Configures the root logger based on application settings."""
    
    log_level_str = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid

    # Check if running in Cloud Run environment (or similar JSON-logging preferred env)
    # Use CLOUD_ENVIRNMENT var set in your Cloud Run config
    use_json_logging = os.environ.get("CLOUD_ENVIRNMENT", "false").lower() == "true"

    # Remove existing handlers to avoid duplicates if called multiple times
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure handler (JSON or plain text)
    handler = logging.StreamHandler(sys.stdout) # Log to stdout

    if use_json_logging:
        # Fields expected by Google Cloud Logging:
        # https://cloud.google.com/logging/docs/structured-logging#special-payload-fields
        # Mapping standard log record attributes to Google Cloud fields.
        format_str = '%(asctime)s %(levelname)s %(name)s %(message)s %(pathname)s %(lineno)d'
        formatter = jsonlogger.JsonFormatter(
            fmt=format_str,
            rename_fields={
                "levelname": "severity", # Map standard levelname to severity
                "asctime": "timestamp", # Use timestamp field
                "name": "logger",
                # Add request ID if captured via context var (requires middleware)
                # 'request_id': 'logging.googleapis.com/trace' # Or use labels
            },
            datefmt='%Y-%m-%dT%H:%M:%S.%fZ' # ISO 8601 format for timestamp
        )
        # Add filter to include request_id if available
        handler.addFilter(RequestIdFilter())
        log_format_type = "JSON (Cloud Run optimized)"
    else:
        # Plain text formatting for local development
        log_format = "[%(asctime)s] [%(levelname)-8s] [%(name)s] [%(request_id)s] %(message)s (%(filename)s:%(lineno)d)"
        date_format = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(log_format, datefmt=date_format)
        # Add filter to include request_id
        handler.addFilter(RequestIdFilter())
        log_format_type = "Plain Text"

    handler.setFormatter(formatter)
    handler.setLevel(log_level)

    # Configure root logger
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Adjust log levels for noisy libraries if needed
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    # Log initial configuration message using the root logger
    root_logger.info(f"Logging configured: Level={log_level_str}, Format={log_format_type}")


# --- Context Filter for Request ID ---
# Requires the request_context_middleware in main.py to set the context var
from contextvars import ContextVar
request_id_var: ContextVar[str] = ContextVar('request_id', default='no-request-id')

class RequestIdFilter(logging.Filter):
    """Injects the request_id from context var into log records."""
    def filter(self, record):
        record.request_id = request_id_var.get()
        return True

# --- Get Logger Function ---
def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger instance."""
    return logging.getLogger(name)

# Example usage (typically called once at startup in main.py)
# if __name__ == "__main__":
#     configure_logging()
#     logger = get_logger(__name__)
#     logger.debug("This is a debug message.")
#     logger.info("This is an info message.")
#     logger.warning("This is a warning message.")
#     logger.error("This is an error message.")
#     try:
#         1 / 0
#     except ZeroDivisionError:
#         logger.exception("This is an exception message.")
