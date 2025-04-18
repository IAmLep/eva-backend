"""
Custom Exception classes for the EVA backend application.
"""

from fastapi import HTTPException, status

# --- Base Exception ---
class AppException(HTTPException):
    """Base exception for custom application errors."""
    def __init__(self, status_code: int, detail: str, headers: dict = None):
        super().__init__(status_code=status_code, detail=detail, headers=headers)

# --- Authentication / Authorization Errors ---
class AuthenticationError(AppException):
    """Exception for authentication failures (invalid credentials, bad token)."""
    def __init__(self, detail: str = "Authentication failed", headers: dict = None):
        # Default headers for bearer token challenges
        final_headers = {"WWW-Authenticate": "Bearer"}
        if headers:
            final_headers.update(headers)
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail, headers=final_headers)

class AuthorizationError(AppException):
    """Exception for authorization failures (insufficient permissions, inactive user)."""
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

# --- Data Validation / Resource Errors ---
class NotFoundException(AppException):
    """Exception when a requested resource is not found."""
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class DuplicateError(AppException):
    """Exception for conflicts, like creating a resource that already exists."""
    def __init__(self, detail: str = "Resource already exists"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)

class BadRequestError(AppException):
    """Exception for general bad requests (invalid input format, missing data)."""
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

# --- Service / Integration Errors ---
class DatabaseError(AppException):
    """Exception for database operation failures."""
    def __init__(self, detail: str = "Database operation failed"):
        # Use 500 for internal DB issues
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

class LLMServiceError(AppException):
    """Exception for failures interacting with the LLM service."""
    def __init__(self, detail: str = "LLM service error"):
        super().__init__(status_code=status.HTTP_502_BAD_GATEWAY, detail=detail) # 502 or 503 might fit

class RateLimitError(AppException):
    """Exception when API rate limits are exceeded."""
    def __init__(self, detail: str = "Rate limit exceeded", headers: dict = None):
        super().__init__(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail, headers=headers)

class ConfigurationError(AppException):
     """Exception for server configuration problems."""
     def __init__(self, detail: str = "Server configuration error"):
          super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)

# --- Specific Application Logic Errors (Add as needed) ---
# class MemoryOperationError(AppException):
#     def __init__(self, detail: str = "Memory operation failed"):
#         super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)
