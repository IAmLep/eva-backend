"""
Security module for EVA backend.

This module provides security-related functionality including authentication
verification, input sanitization, CSRF protection, and security headers.

Last updated: 2025-04-01 11:09:55
Version: v1.8.6
Created by: IAmLep
"""

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set, Tuple, Union, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, status
from fastapi.security import APIKeyCookie, APIKeyHeader, APIKeyQuery, OAuth2PasswordBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from auth import get_current_user
from config import get_settings
from exceptions import AuthenticationError, AuthorizationError
from logging_config import get_logger
from models import User, ApiKeyScope

# Logger configuration
logger = get_logger(__name__)


# API key security schemes
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# OAuth2 password bearer scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)

# CSRF token cookie name
CSRF_TOKEN_COOKIE = "eva_csrf_token"
CSRF_TOKEN_HEADER = "X-CSRF-Token"


class SecurityConfig:
    """
    Security configuration for the application.
    
    Centralizes security-related configuration settings and provides
    methods for common security operations.
    """
    
    def __init__(self):
        """Initialize security configuration with settings."""
        self.settings = get_settings()
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache"
        }
        
        # Content security policy
        self.csp = {
            "default-src": ["'self'"],
            "img-src": ["'self'", "data:", "https://storage.googleapis.com"],
            "style-src": ["'self'", "'unsafe-inline'"],
            "script-src": ["'self'"],
            "connect-src": ["'self'", "https://api.openai.com", "https://generativelanguage.googleapis.com"],
            "frame-ancestors": ["'none'"],
            "form-action": ["'self'"],
            "base-uri": ["'self'"],
            "object-src": ["'none'"]
        }
        
        # Rate limiting configuration is loaded from settings
        
        # Allowed content types for API requests
        self.allowed_content_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data"
        }
        
        # Patterns for sanitizing inputs
        self.xss_pattern = re.compile(r'<script.*?>.*?</script>', re.IGNORECASE | re.DOTALL)
        
        logger.info("Security configuration initialized")
    
    def get_csp_header(self) -> str:
        """
        Generate Content Security Policy header value.
        
        Returns:
            str: Formatted CSP header value
        """
        csp_parts = []
        for directive, sources in self.csp.items():
            csp_parts.append(f"{directive} {' '.join(sources)}")
        
        return "; ".join(csp_parts)
    
    def apply_security_headers(self, response: Response) -> None:
        """
        Apply security headers to response.
        
        Args:
            response: Response to modify
        """
        # Add basic security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        # Add CSP header
        response.headers["Content-Security-Policy"] = self.get_csp_header()
    
    def generate_csrf_token(self) -> str:
        """
        Generate a secure CSRF token.
        
        Returns:
            str: Generated CSRF token
        """
        return secrets.token_hex(32)
    
    def verify_csrf_token(self, request_token: str, stored_token: str) -> bool:
        """
        Verify CSRF token.
        
        Args:
            request_token: Token from request
            stored_token: Token from session or cookie
            
        Returns:
            bool: True if token is valid
        """
        if not request_token or not stored_token:
            return False
        
        return hmac.compare_digest(request_token, stored_token)
    
    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input text to prevent XSS attacks.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            str: Sanitized text
        """
        if not text:
            return text
        
        # Remove script tags
        sanitized = self.xss_pattern.sub('', text)
        
        # Other sanitization could be added here
        
        return sanitized
    
    def generate_nonce(self) -> str:
        """
        Generate a nonce for CSP inline scripts.
        
        Returns:
            str: Generated nonce
        """
        return base64.b64encode(os.urandom(16)).decode('utf-8')


# Singleton instance
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """
    Get security configuration singleton.
    
    Returns:
        SecurityConfig: Security configuration instance
    """
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware for FastAPI.
    
    Applies security headers, CSRF protection, and request validation.
    """
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request through security middleware.
        
        Args:
            request: FastAPI request
            call_next: Next middleware or route handler
            
        Returns:
            Response: HTTP response
        """
        # Get security configuration
        security_config = get_security_config()
        
        # Check if request method is safe (GET, HEAD, OPTIONS)
        is_safe_method = request.method in ("GET", "HEAD", "OPTIONS")
        
        # For non-safe methods, validate CSRF token unless it's API route
        if not is_safe_method and not request.url.path.startswith("/api/v1/"):
            # Get CSRF token from cookies and headers
            csrf_cookie = request.cookies.get(CSRF_TOKEN_COOKIE)
            csrf_header = request.headers.get(CSRF_TOKEN_HEADER)
            
            # Validate CSRF token
            if not security_config.verify_csrf_token(csrf_header, csrf_cookie):
                logger.warning(f"CSRF validation failed for request to {request.url.path}")
                return Response(
                    content=json.dumps({
                        "detail": "CSRF token verification failed",
                        "code": "csrf_verification_error"
                    }),
                    status_code=status.HTTP_403_FORBIDDEN,
                    media_type="application/json"
                )
        
        # Process the request
        response = await call_next(request)
        
        # Generate new CSRF token for responses
        if isinstance(response, Response) and not request.url.path.startswith("/api/v1/"):
            csrf_token = security_config.generate_csrf_token()
            response.set_cookie(
                key=CSRF_TOKEN_COOKIE,
                value=csrf_token,
                httponly=True,
                secure=not security_config.settings.DEBUG,
                samesite="lax"
            )
        
        # Apply security headers
        if isinstance(response, Response):
            security_config.apply_security_headers(response)
        
        return response


async def get_api_key(
    api_key_header: str = Security(API_KEY_HEADER),
    api_key_query: str = Security(API_KEY_QUERY)
) -> Optional[str]:
    """
    Get API key from request.
    
    Args:
        api_key_header: API key from header
        api_key_query: API key from query parameter
        
    Returns:
        Optional[str]: API key if found, None otherwise
    """
    return api_key_header or api_key_query


async def validate_api_key(api_key: str = Depends(get_api_key)) -> User:
    """
    Validate API key and get associated user.
    
    Args:
        api_key: API key to validate
        
    Returns:
        User: User associated with API key
        
    Raises:
        AuthenticationError: If API key is invalid
    """
    if not api_key:
        logger.warning("API key authentication attempted with no key provided")
        raise AuthenticationError(detail="API key required")
    
    # In a real implementation, this would check against database
    # For now, we'll use a placeholder
    db = get_db_manager()
    
    # Hash the API key for comparison
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Look up the API key
    key_record = await db.get_api_key_by_hash(hashed_key)
    
    if not key_record:
        logger.warning(f"Invalid API key used: {api_key[:5]}...")
        raise AuthenticationError(detail="Invalid API key")
    
    # Check if key is expired
    if key_record.expires_at and key_record.expires_at < datetime.utcnow():
        logger.warning(f"Expired API key used: {api_key[:5]}...")
        raise AuthenticationError(detail="API key expired")
    
    # Check if key is active
    if not key_record.is_active:
        logger.warning(f"Inactive API key used: {api_key[:5]}...")
        raise AuthenticationError(detail="API key is inactive")
    
    # Update last used timestamp
    await db.update_api_key_usage(key_record.key_id)
    
    # Get user associated with the key
    user = await db.get_user_by_id(key_record.user_id)
    
    if not user:
        logger.error(f"API key {api_key[:5]}... associated with non-existent user: {key_record.user_id}")
        raise AuthenticationError(detail="User not found")
    
    # Check if user is disabled
    if user.disabled:
        logger.warning(f"API key {api_key[:5]}... used by disabled user: {user.username}")
        raise AuthenticationError(detail="User is disabled")
    
    logger.info(f"API key authentication successful for user: {user.username}")
    
    # Store API key info in request state for later use in authorization
    # This will be used by require_api_scope
    request = get_current_request()
    if request:
        request.state.api_key_scopes = key_record.scopes
    
    return user


def require_api_scope(required_scope: ApiKeyScope):
    """
    Dependency to require specific API key scope.
    
    Args:
        required_scope: Scope required for the operation
        
    Returns:
        Callable: FastAPI dependency
    """
    async def dependency(request: Request, user: User = Depends(validate_api_key)) -> None:
        """
        Check if API key has required scope.
        
        Args:
            request: FastAPI request
            user: Authenticated user
            
        Raises:
            AuthorizationError: If API key lacks required scope
        """
        # Get scopes from request state (set by validate_api_key)
        scopes = getattr(request.state, "api_key_scopes", [])
        
        # Check if required scope is present
        # Note: ADMIN scope includes all permissions
        if required_scope not in scopes and ApiKeyScope.ADMIN not in scopes:
            logger.warning(
                f"API key used by {user.username} lacks required scope: {required_scope}"
            )
            raise AuthorizationError(detail=f"API key requires {required_scope} scope")
    
    return dependency


def setup_security(app: FastAPI) -> None:
    """
    Set up security for FastAPI application.
    
    Args:
        app: FastAPI application
    """
    settings = get_settings()
    
    # Add security middleware
    app.add_middleware(SecurityMiddleware)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )
    
    logger.info("Security setup complete")


def get_current_request() -> Optional[Request]:
    """
    Get current request from context.
    
    Returns:
        Optional[Request]: Current request if available
    """
    # In a real implementation, this would use contextvars
    # For now, return None as placeholder
    return None


def sanitize_all_inputs(data: Any) -> Any:
    """
    Recursively sanitize all string inputs in data structure.
    
    Args:
        data: Input data to sanitize
        
    Returns:
        Any: Sanitized data
    """
    security_config = get_security_config()
    
    if isinstance(data, str):
        return security_config.sanitize_input(data)
    elif isinstance(data, dict):
        return {k: sanitize_all_inputs(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_all_inputs(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_all_inputs(item) for item in data)
    else:
        return data


def generate_secure_password(length: int = 16) -> str:
    """
    Generate a cryptographically secure random password.
    
    Args:
        length: Password length
        
    Returns:
        str: Generated password
    """
    # Define character sets
    uppercase = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # Omit I and O (to avoid confusion with 1 and 0)
    lowercase = "abcdefghijkmnpqrstuvwxyz"  # Omit l (to avoid confusion with 1)
    numbers = "23456789"  # Omit 0 and 1 (to avoid confusion with O and l)
    special = "!@#$%^&*()-_=+[]{}|;:,.<>?"
    
    # Ensure at least one character from each set
    password = [
        secrets.choice(uppercase),
        secrets.choice(lowercase),
        secrets.choice(numbers),
        secrets.choice(special)
    ]
    
    # Fill the rest with random characters from all sets
    all_chars = uppercase + lowercase + numbers + special
    password.extend(secrets.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password
    secrets.SystemRandom().shuffle(password)
    
    return ''.join(password)


def hash_sensitive_data(data: str) -> str:
    """
    Hash sensitive data using SHA-256.
    
    Args:
        data: Data to hash
        
    Returns:
        str: Hashed data
    """
    return hashlib.sha256(data.encode()).hexdigest()


class APIKeyManager:
    """
    API key management utility.
    
    Handles creation, validation, and management of API keys.
    """
    
    def __init__(self):
        """Initialize API key manager."""
        self.settings = get_settings()
    
    def generate_api_key(self) -> Tuple[str, str]:
        """
        Generate a new API key and prefix.
        
        Returns:
            Tuple[str, str]: API key and its prefix
        """
        # Generate a random API key
        key = secrets.token_hex(32)
        
        # Create a prefix for reference (first 8 characters)
        prefix = key[:8]
        
        return key, prefix
    
    def hash_api_key(self, key: str) -> str:
        """
        Hash an API key for storage.
        
        Args:
            key: API key to hash
            
        Returns:
            str: Hashed API key
        """
        return hashlib.sha256(key.encode()).hexdigest()
    
    def validate_api_key_format(self, key: str) -> bool:
        """
        Validate API key format.
        
        Args:
            key: API key to validate
            
        Returns:
            bool: True if format is valid
        """
        # Check length
        if len(key) != 64:
            return False
        
        # Check if key contains only hexadecimal characters
        try:
            int(key, 16)
            return True
        except ValueError:
            return False


# Singleton instance
_api_key_manager: Optional[APIKeyManager] = None


def get_api_key_manager() -> APIKeyManager:
    """
    Get API key manager singleton.
    
    Returns:
        APIKeyManager: API key manager instance
    """
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager