"""
Error Middleware module for EVA backend.

This module provides global error handling middleware to ensure
consistent error responses throughout the application.


Version 3 working
"""

import logging
import traceback
from typing import Callable, Dict, List, Optional, Union, Any

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from starlette.middleware.base import BaseHTTPMiddleware

from exceptions import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    CustomException,
    DatabaseError,
    DuplicateError,
    LLMServiceError,
    NotFoundException,
    RateLimitError,
    SyncError,
    ValidationError,
    WebSocketError,
)

# Logger configuration
logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Global error handling middleware for FastAPI.
    
    This middleware catches all unhandled exceptions and converts them
    to standardized JSON responses with appropriate status codes.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process a request and handle any exceptions.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response: HTTP response
        """
        try:
            # Process the request normally
            return await call_next(request)
            
        except CustomException as exc:
            # Handle our custom exceptions
            return self._handle_custom_exception(request, exc)
            
        except PydanticValidationError as exc:
            # Handle Pydantic validation errors
            return self._handle_validation_error(request, exc)
            
        except Exception as exc:
            # Handle unexpected exceptions
            return self._handle_unexpected_exception(request, exc)
    
    def _handle_custom_exception(
        self, request: Request, exc: CustomException
    ) -> JSONResponse:
        """
        Handle custom application exceptions.
        
        Args:
            request: FastAPI request
            exc: Custom exception instance
            
        Returns:
            JSONResponse: Formatted error response
        """
        # Log different error levels based on status code
        log_func = self._get_log_function(exc.status_code)
        log_func(
            f"Custom exception: {exc.detail} "
            f"(code: {exc.code}, status: {exc.status_code}, "
            f"path: {request.url.path})"
        )
        
        # Convert exception to response
        content = {
            "detail": exc.detail,
            "code": exc.code,
            "path": request.url.path,
        }
        
        # Add field errors for validation errors
        if isinstance(exc, ValidationError) and exc.field_errors:
            content["field_errors"] = exc.field_errors
        
        # Add reset_at for rate limit errors
        if isinstance(exc, RateLimitError) and exc.reset_at:
            content["reset_at"] = exc.reset_at
        
        # Add conflict items for sync errors
        if isinstance(exc, SyncError) and exc.conflict_items:
            content["conflicts"] = exc.conflict_items
            
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=exc.headers
        )
    
    def _handle_validation_error(
        self, request: Request, exc: PydanticValidationError
    ) -> JSONResponse:
        """
        Handle Pydantic validation errors.
        
        Args:
            request: FastAPI request
            exc: Pydantic validation error
            
        Returns:
            JSONResponse: Formatted error response
        """
        # Convert Pydantic errors to our format
        field_errors = {}
        
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error["msg"])
        
        logger.warning(
            f"Validation error: {str(exc)} "
            f"(path: {request.url.path})"
        )
        
        # Create our custom validation error
        validation_error = ValidationError(
            detail="Request validation failed",
            field_errors=field_errors
        )
        
        # Return formatted response
        return JSONResponse(
            status_code=validation_error.status_code,
            content={
                "detail": validation_error.detail,
                "code": validation_error.code,
                "field_errors": field_errors,
                "path": request.url.path,
            }
        )
    
    def _handle_unexpected_exception(
        self, request: Request, exc: Exception
    ) -> JSONResponse:
        """
        Handle unexpected exceptions.
        
        Args:
            request: FastAPI request
            exc: Unexpected exception
            
        Returns:
            JSONResponse: Formatted error response
        """
        # Get traceback for unexpected errors
        error_traceback = traceback.format_exc()
        
        # Log full error details
        logger.error(
            f"Unexpected error: {str(exc)} "
            f"(path: {request.url.path})\n{error_traceback}"
        )
        
        # In production, don't expose internal error details
        from config import get_settings
        settings = get_settings()
        
        if settings.is_production:
            detail = "An unexpected error occurred"
        else:
            detail = str(exc)
        
        # Return formatted response
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": detail,
                "code": "internal_server_error",
                "path": request.url.path,
            }
        )
    
    def _get_log_function(self, status_code: int) -> Callable:
        """
        Get appropriate logger function based on status code.
        
        Args:
            status_code: HTTP status code
            
        Returns:
            Callable: Logging function
        """
        if status_code >= 500:
            return logger.error
        elif status_code >= 400:
            return logger.warning
        else:
            return logger.info


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register exception handlers for a FastAPI application.
    
    This function sets up specific handlers for known exception types
    that might be raised by endpoint functions directly.
    
    Args:
        app: FastAPI application instance
    """
    @app.exception_handler(CustomException)
    async def custom_exception_handler(request: Request, exc: CustomException):
        # Log different error levels based on status code
        log_func = logger.error if exc.status_code >= 500 else logger.warning
        log_func(
            f"Endpoint exception: {exc.detail} "
            f"(code: {exc.code}, status: {exc.status_code}, "
            f"path: {request.url.path})"
        )
        
        # Convert exception to response
        content = {
            "detail": exc.detail,
            "code": exc.code,
            "path": request.url.path,
        }
        
        # Add specific fields for certain exception types
        if isinstance(exc, ValidationError) and exc.field_errors:
            content["field_errors"] = exc.field_errors
        
        if isinstance(exc, RateLimitError) and exc.reset_at:
            content["reset_at"] = exc.reset_at
        
        if isinstance(exc, SyncError) and exc.conflict_items:
            content["conflicts"] = exc.conflict_items
            
        return JSONResponse(
            status_code=exc.status_code,
            content=content,
            headers=exc.headers
        )
    
    @app.exception_handler(PydanticValidationError)
    async def validation_exception_handler(request: Request, exc: PydanticValidationError):
        # Process the same way as in the middleware
        field_errors = {}
        
        for error in exc.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error["msg"])
        
        logger.warning(
            f"Validation error in endpoint: {str(exc)} "
            f"(path: {request.url.path})"
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": "Request validation failed",
                "code": "validation_error",
                "field_errors": field_errors,
                "path": request.url.path,
            }
        )
    
    @app.exception_handler(Exception)
    async def unexpected_exception_handler(request: Request, exc: Exception):
        # Process the same way as in the middleware
        error_traceback = traceback.format_exc()
        
        logger.error(
            f"Unexpected error in endpoint: {str(exc)} "
            f"(path: {request.url.path})\n{error_traceback}"
        )
        
        from config import get_settings
        settings = get_settings()
        
        if settings.is_production:
            detail = "An unexpected error occurred"
        else:
            detail = str(exc)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": detail,
                "code": "internal_server_error",
                "path": request.url.path,
            }
        )