"""
Global Error Handling Middleware for the FastAPI application.

Catches specified exceptions and returns standardized JSON error responses.
"""

import logging

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# Import custom exceptions
from exceptions import AppException, AuthenticationError, AuthorizationError, NotFoundException, DuplicateError, BadRequestError, DatabaseError, LLMServiceError, RateLimitError, ConfigurationError

logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to catch exceptions and return standard JSON errors."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            # Try processing the request
            response = await call_next(request)
            return response
        except Exception as exc:
            # Get request ID if available from context middleware
            request_id = getattr(request.state, "request_id", "N/A")
            error_id = f"ERR-{request_id}" # Create a unique error ID

            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            detail = "An unexpected internal server error occurred."
            headers = {}

            # Handle specific custom exceptions first
            if isinstance(exc, AppException):
                status_code = exc.status_code
                detail = exc.detail
                headers = exc.headers or {}
                log_level = logging.WARNING if status_code < 500 else logging.ERROR
                logger.log(log_level, f"AppException caught: {type(exc).__name__} ({status_code}) - {detail} (Req ID: {request_id})", exc_info=False) # Don't need full stack trace for handled AppExceptions

            # Handle other common exceptions (e.g., Pydantic validation errors)
            # FastAPI handles RequestValidationError by default, but we could customize it here if needed.

            # Handle generic exceptions last
            else:
                # Log the full exception details for unexpected errors
                logger.exception(f"Unhandled exception caught: {type(exc).__name__} - {exc} (Error ID: {error_id}, Req ID: {request_id})", exc_info=exc)
                # Keep detail generic for unexpected errors sent to client
                detail = f"Internal Server Error. Please report this issue with Error ID: {error_id}"

            # Construct and return JSON error response
            error_content = {"detail": detail, "error_id": error_id}
            return JSONResponse(
                status_code=status_code,
                content=error_content,
                headers=headers,
            )
