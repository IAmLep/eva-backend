"""
Exceptions module for EVA backend.

This module provides custom exception classes for consistent
error handling throughout the application.

Last updated: 2025-04-01 10:28:09
Version: v1.8.6
Created by: IAmLep
"""

from typing import Dict, List, Optional, Union, Any


class CustomException(Exception):
    """
    Base custom exception class.
    
    All custom exceptions in the application inherit from this base class
    to ensure consistent error handling and response formatting.
    
    Attributes:
        detail: Human-readable error description
        code: Machine-readable error code
        status_code: HTTP status code
        headers: Optional response headers
    """
    def __init__(
        self, 
        detail: str, 
        code: str = "internal_error",
        status_code: int = 500,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize custom exception.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            status_code: HTTP status code
            headers: Optional response headers
        """
        self.detail = detail
        self.code = code
        self.status_code = status_code
        self.headers = headers or {}
        super().__init__(self.detail)


class AuthenticationError(CustomException):
    """
    Authentication-related errors.
    
    Raised when authentication fails or credentials are invalid.
    
    Examples:
        - Invalid JWT token
        - Expired credentials
        - Missing authentication headers
    """
    def __init__(
        self, 
        detail: str = "Authentication failed", 
        code: str = "authentication_error",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize authentication error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=401,
            headers=headers or {"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(CustomException):
    """
    Authorization-related errors.
    
    Raised when user does not have permission to access a resource.
    
    Examples:
        - Insufficient permissions
        - Access to another user's resources
        - Disabled account
    """
    def __init__(
        self, 
        detail: str = "Not authorized", 
        code: str = "authorization_error",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize authorization error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=403,
            headers=headers
        )


class NotFoundException(CustomException):
    """
    Resource not found errors.
    
    Raised when a requested resource does not exist.
    
    Examples:
        - User not found
        - Memory not found
        - Configuration not found
    """
    def __init__(
        self, 
        detail: str = "Resource not found", 
        code: str = "not_found",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize not found error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=404,
            headers=headers
        )


class ValidationError(CustomException):
    """
    Data validation errors.
    
    Raised when input data fails validation.
    
    Examples:
        - Invalid email format
        - Required field missing
        - Value out of allowed range
    """
    def __init__(
        self, 
        detail: str = "Validation error", 
        code: str = "validation_error",
        field_errors: Optional[Dict[str, List[str]]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize validation error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            field_errors: Optional mapping of field names to error messages
            headers: Optional response headers
        """
        self.field_errors = field_errors or {}
        super().__init__(
            detail=detail,
            code=code,
            status_code=422,
            headers=headers
        )


class DatabaseError(CustomException):
    """
    Database-related errors.
    
    Raised when database operations fail.
    
    Examples:
        - Connection failure
        - Query execution error
        - Constraint violation
    """
    def __init__(
        self, 
        detail: str = "Database error", 
        code: str = "database_error",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize database error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=500,
            headers=headers
        )


class RateLimitError(CustomException):
    """
    Rate limit exceeded errors.
    
    Raised when a client exceeds allowed request rate.
    
    Examples:
        - Too many requests in a time period
        - API quota exceeded
    """
    def __init__(
        self, 
        detail: str = "Rate limit exceeded", 
        code: str = "rate_limit_exceeded",
        reset_at: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize rate limit error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            reset_at: Optional timestamp when rate limit resets
            headers: Optional response headers
        """
        self.reset_at = reset_at
        headers = headers or {}
        
        if reset_at:
            headers["Retry-After"] = reset_at
            
        super().__init__(
            detail=detail,
            code=code,
            status_code=429,
            headers=headers
        )


class LLMServiceError(CustomException):
    """
    LLM service-related errors.
    
    Raised when communication with external LLM services fails.
    
    Examples:
        - Gemini API error
        - Context length exceeded
        - Invalid prompt
    """
    def __init__(
        self, 
        detail: str = "LLM service error", 
        code: str = "llm_service_error",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize LLM service error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=502,
            headers=headers
        )


class ConfigurationError(CustomException):
    """
    Configuration-related errors.
    
    Raised when there are issues with application configuration.
    
    Examples:
        - Missing required environment variable
        - Invalid configuration value
        - Incompatible settings
    """
    def __init__(
        self, 
        detail: str = "Configuration error", 
        code: str = "configuration_error",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize configuration error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=500,
            headers=headers
        )


class WebSocketError(CustomException):
    """
    WebSocket-related errors.
    
    Raised when WebSocket operations fail.
    
    Examples:
        - Connection failure
        - Message format error
        - Protocol violation
    """
    def __init__(
        self, 
        detail: str = "WebSocket error", 
        code: str = "websocket_error",
        close_code: int = 1008,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize WebSocket error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            close_code: WebSocket close code
            headers: Optional response headers
        """
        self.close_code = close_code
        super().__init__(
            detail=detail,
            code=code,
            status_code=500,
            headers=headers
        )


class DuplicateError(CustomException):
    """
    Duplicate resource errors.
    
    Raised when trying to create a resource that already exists.
    
    Examples:
        - Username already taken
        - Email already registered
        - Duplicate record
    """
    def __init__(
        self, 
        detail: str = "Resource already exists", 
        code: str = "duplicate_error",
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize duplicate error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            headers: Optional response headers
        """
        super().__init__(
            detail=detail,
            code=code,
            status_code=409,
            headers=headers
        )


class SyncError(CustomException):
    """
    Synchronization-related errors.
    
    Raised when data synchronization operations fail.
    
    Examples:
        - Conflict during merge
        - Inconsistent state
        - Version mismatch
    """
    def __init__(
        self, 
        detail: str = "Synchronization error", 
        code: str = "sync_error",
        conflict_items: Optional[List[Dict[str, Any]]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialize synchronization error.
        
        Args:
            detail: Human-readable error description
            code: Machine-readable error code
            conflict_items: Optional list of conflicting items
            headers: Optional response headers
        """
        self.conflict_items = conflict_items or []
        super().__init__(
            detail=detail,
            code=code,
            status_code=409,
            headers=headers
        )