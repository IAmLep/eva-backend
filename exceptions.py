from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional

class GeminiAPIError(HTTPException):
    """Raised when there is an error interacting with the Gemini API."""
    def __init__(self, detail: str = "Gemini API error", status_code: int = 500):
        super().__init__(status_code=status_code, detail=detail)

class MemoryExtractionError(HTTPException):
    """Raised when there is an error extracting memory from conversation history."""
    def __init__(self, detail: str = "Memory extraction failed", status_code: int = 500):
        super().__init__(status_code=status_code, detail=detail)

class AuthenticationError(HTTPException):
    """Raised when authentication fails."""
    def __init__(self, detail: str = "Not authenticated", status_code: int = 401):
        super().__init__(status_code=status_code, detail=detail)

class DatabaseError(HTTPException):
    """Raised when there is a database error."""
    def __init__(self, detail: str = "Database operation failed"):
        super().__init__(status_code=500, detail=detail)

class BaseEvaException(Exception):
    """Base exception for all Eva-specific exceptions"""
    code: str = "internal_error"
    status_code: int = 500
    message: str = "An unexpected error occurred"
    
    def __init__(self, message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message or self.message
        self.details = details or {}
        super().__init__(self.message)

class LLMServiceException(BaseEvaException):
    code = "llm_service_error"
    status_code = 503
    message = "LLM service is currently unavailable"

class DatabaseException(BaseEvaException):
    code = "database_error"
    status_code = 503
    message = "Database operation failed"

class ToolAPIException(BaseEvaException):
    code = "tool_api_error"
    status_code = 503
    message = "External tool API is unavailable"
