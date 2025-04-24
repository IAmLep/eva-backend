"""
Security module for EVA backend.

Handles API key authentication, Google ID‑token & internal JWT (HS256) bearer tokens,
scope checking, and sets security headers (CSP, HSTS, etc.).
"""

import logging
import secrets
import asyncio
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import (
    APIKeyHeader,
    APIKeyQuery,
    HTTPBearer,
    HTTPAuthorizationCredentials
)
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware

import jwt
from jwt import PyJWTError

# Google ID‑token verification
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

from config import settings
from database import get_db_manager, DatabaseManager
from exceptions import AuthenticationError, AuthorizationError
from models import User, ApiKey, ApiKeyScope

logger = logging.getLogger(__name__)

# --- Security Headers Middleware ---
class SecurityConfig:
    def __init__(self):
        self.settings = settings
        self.hsts_enabled = getattr(self.settings, "is_production", False)
        self.hsts_max_age = 31536000

    def apply_security_headers(self, response: Response) -> None:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if self.hsts_enabled:
            response.headers[
                "Strict-Transport-Security"
            ] = f"max-age={self.hsts_max_age}; includeSubDomains"

_security_config: Optional[SecurityConfig] = None
def get_security_config() -> SecurityConfig:
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        get_security_config().apply_security_headers(response)
        return response

def setup_security(app: FastAPI) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware applied.")

# --- API Key Scheme ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)
api_key_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class APIKeyManager:
    PREFIX_LENGTH = 8
    SECRET_LENGTH = 32

    def generate_api_key(self) -> Tuple[str, str, str]:
        prefix = secrets.token_urlsafe(self.PREFIX_LENGTH)
        secret_part = secrets.token_urlsafe(self.SECRET_LENGTH)
        full_key = f"{prefix}_{secret_part}"
        hashed_key = api_key_context.hash(full_key)
        return full_key, hashed_key, prefix

    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        try:
            return api_key_context.verify(plain_key, hashed_key)
        except Exception:
            return False

    def validate_api_key_format(self, key: str) -> bool:
        parts = key.split('_')
        return len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0

_api_key_manager: Optional[APIKeyManager] = None
def get_api_key_manager() -> APIKeyManager:
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager

async def get_api_key(
    api_key_query: Annotated[Optional[str], Depends(API_KEY_QUERY)] = None,
    api_key_header: Annotated[Optional[str], Depends(API_KEY_HEADER)] = None,
) -> Optional[str]:
    return api_key_header or api_key_query

async def validate_api_key(
    request: Request,
    api_key: Annotated[str, Depends(get_api_key)],
    db: DatabaseManager = Depends(get_db_manager),
    key_manager: APIKeyManager = Depends(get_api_key_manager)
) -> User:
    if not api_key:
        raise AuthenticationError(detail="API key required")

    if not key_manager.validate_api_key_format(api_key):
        raise AuthenticationError(detail="Invalid API key format")

    prefix = api_key.split('_', 1)[0]
    candidates = await db.get_api_keys_by_prefix(prefix)
    validated: Optional[ApiKey] = None
    for key_data in candidates:
        if key_manager.verify_api_key(api_key, key_data.hashed_key):
            validated = key_data
            break

    if not validated:
        logger.warning("API key authentication failed for prefix %s", prefix)
        raise AuthenticationError(detail="Invalid API key")

    if not validated.is_active or (validated.expires_at and validated.expires_at < datetime.now(timezone.utc)):
        raise AuthenticationError(detail="API key inactive or expired")

    user = await db.get_user(validated.user_id)
    if not user or user.disabled:
        raise AuthenticationError(detail="User not found or disabled")

    request.state.validated_api_key = validated
    # update last used asynchronously
    asyncio.create_task(db.update_api_key_usage(validated.key_id))

    return user

# --- Bearer Token Scheme (HS256 or Google ID‑token RS256) ---
bearer_scheme = HTTPBearer(auto_error=False)

def verify_token(token: str) -> Dict[str, Any]:
    # Try internal HS256 first
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            audience=getattr(settings, "API_AUDIENCE", None),
        )
        logger.debug("Internal JWT validated")
        return payload
    except PyJWTError as e:
        logger.debug("HS256 decode failed (%s), trying Google ID‑token", e)

    # Fallback to Google ID‑token (RS256)
    try:
        req = google_requests.Request()
        aud = settings.BACKEND_URL.rstrip('/')
        payload = google_id_token.verify_oauth2_token(token, req, aud)
        logger.debug("Google ID‑token validated for aud=%s", aud)
        return payload
    except Exception as e:
        logger.warning("Google ID‑token validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def validate_bearer(
    request: Request,
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)]
) -> User:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    data = verify_token(credentials.credentials)
    # determine user identity from payload
    user_id = data.get("sub") or data.get("user_id") or data.get("email")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # load your User by ID or email
    try:
        user = User.get_by_id_or_email(user_id)
    except Exception:
        logger.exception("User lookup failed")
        raise HTTPException(status_code=401, detail="Invalid user")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User is inactive")

    return user

# --- Combined Dependency ###
async def get_current_user(
    request: Request,
    api_user: Optional[User] = Depends(validate_api_key),
    bearer_user: Optional[User] = Depends(validate_bearer),
) -> User:
    """
    Try API key first; if that fails, try bearer token.
    """
    # FastAPI will run both dependencies. If API key is valid, return that user.
    if api_user:
        return api_user
    # otherwise fallback to bearer_user
    return bearer_user