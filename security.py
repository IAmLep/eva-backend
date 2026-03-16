"""
Security module for EVA backend.

Provides security headers middleware and API key validation utilities.

=============================================================================
AUTH NOTE - LEGACY AUTHENTICATION FUNCTIONS
=============================================================================
The verify_token(), validate_bearer(), and get_current_user() functions in
this file are LEGACY / UNUSED duplicates of the primary auth in auth.py.

The intended authentication flow for EVA is:
  1. Frontend: Google sign-in via Firebase Auth (firebase_auth.py)
  2. Frontend sends Firebase ID token to POST /api/v1/auth/firebase
  3. Backend verifies token, returns internal HS256 JWT (auth.py)
  4. All subsequent API calls use the internal JWT via auth.py's
     get_current_user() / get_current_active_user()

The functions below (verify_token, validate_bearer, get_current_user) are
kept for reference but are NOT wired into any router. Do not add new
dependencies on them. Use auth.py instead.

The SecurityHeadersMiddleware and setup_security() ARE actively used.
=============================================================================
"""
import logging
import secrets
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import jwt
from jwt import PyJWTError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings
from database import get_db_manager, DatabaseManager
from exceptions import AuthenticationError, AuthorizationError
from models import User, ApiKey, ApiKeyScope

logger = logging.getLogger(__name__)
settings = get_settings()

# --- API Key Schemes ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# --- LEGACY: Bearer Token Verification ---
# These are NOT used by any router. auth.py's get_current_user is the
# active auth dependency. Kept for reference only.
bearer_scheme = HTTPBearer(auto_error=False)

def verify_token(token: str) -> Dict[str, Any]:
    """
    LEGACY - NOT USED. See auth.py for active token verification.

    Verifies the provided token using either HS256 (internal) or RS256 (Google ID token).

    Args:
        token (str): The JWT token to verify.

    Returns:
        Dict[str, Any]: The decoded token payload.

    Raises:
        HTTPException: If the token cannot be verified.
    """
    # 1. Attempt HS256 verification (internal token)
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=["HS256"],  # Restrict to HS256
            audience=settings.API_AUDIENCE,
        )
        logger.debug("Internal JWT validated successfully using HS256.")
        return payload
    except PyJWTError as e:
        logger.debug(f"HS256 validation failed: {e}. Falling back to RS256 validation.")

    # 2. Attempt RS256 verification (Google ID token)
    try:
        req = google_requests.Request()
        payload = google_id_token.verify_oauth2_token(
            token,
            req,
            settings.BACKEND_URL
        )
        logger.debug("Google ID token validated successfully using RS256.")
        return payload
    except Exception as e:
        logger.warning(f"RS256 validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def validate_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> User:
    """
    LEGACY - NOT USED. See auth.py for active bearer validation.

    Validates a bearer token and retrieves the associated user.

    Args:
        credentials (Optional[HTTPAuthorizationCredentials]): The bearer token credentials.

    Returns:
        User: The associated user.

    Raises:
        HTTPException: If the token is missing or invalid.
    """
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token missing")

    token_data = verify_token(credentials.credentials)
    user_id = token_data.get("sub") or token_data.get("user_id") or token_data.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Fetch user from the database
    user = await get_db_manager().get_user(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive or unknown user")

    return user

# --- LEGACY: Combined User Dependency ---
async def get_current_user(
    bearer_user: Optional[User] = Depends(validate_bearer),
) -> User:
    """
    LEGACY - NOT USED. See auth.py's get_current_user() for the active
    auth dependency used by all routers.

    Args:
        bearer_user (Optional[User]): The bearer-authenticated user.

    Returns:
        User: The authenticated user.
    """
    return bearer_user

# --- Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Adds security-related headers to each response.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

def setup_security(app):
    """
    Configures security-related functionalities for the FastAPI app.

    Args:
        app (FastAPI): The FastAPI application instance.
    """
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware applied.")