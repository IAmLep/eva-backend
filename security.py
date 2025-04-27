"""
Security module for EVA backend.

Supports:
 - X‑API‑Key via header/query + scope checks
 - Internal HS256 JWT access tokens
 - Google RS256 ID‑tokens (Cloud Run audience)
 - Security headers (CSP, HSTS, etc.)
"""
import logging
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import jwt
from jwt import PyJWTError
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    APIKeyHeader,
    APIKeyQuery,
    HTTPBearer,
    HTTPAuthorizationCredentials
)
from passlib.context import CryptContext
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
from starlette.middleware.base import BaseHTTPMiddleware

from config import get_settings
from database import get_db_manager, DatabaseManager
from exceptions import AuthenticationError, AuthorizationError
from models import User, ApiKey, ApiKeyScope

logger = logging.getLogger(__name__)
settings = get_settings()

# --- API‑Key Schemes & Manager ---
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class APIKeyManager:
    PREFIX_LENGTH = 8
    SECRET_LENGTH = 32

    def generate_api_key(self) -> Tuple[str, str, str]:
        prefix = secrets.token_urlsafe(self.PREFIX_LENGTH)
        secret = secrets.token_urlsafe(self.SECRET_LENGTH)
        full_key = f"{prefix}_{secret}"
        hashed = pwd_context.hash(full_key)
        return full_key, hashed, prefix

    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        try:
            return pwd_context.verify(plain_key, hashed_key)
        except Exception:
            return False

    def validate_api_key_format(self, key: str) -> bool:
        parts = key.split('_')
        return len(parts) == 2 and all(parts)

_api_key_manager: Optional[APIKeyManager] = None
def get_api_key_manager() -> APIKeyManager:
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager

async def get_api_key(
    api_key_query: Optional[str] = Depends(API_KEY_QUERY),
    api_key_header: Optional[str] = Depends(API_KEY_HEADER),
) -> Optional[str]:
    return api_key_header or api_key_query

async def validate_api_key(
    request: Request,
    api_key: str = Depends(get_api_key),
    db: DatabaseManager = Depends(get_db_manager),
    key_manager: APIKeyManager = Depends(get_api_key_manager),
) -> User:
    if not api_key:
        raise AuthenticationError(detail="API key required")
    if not key_manager.validate_api_key_format(api_key):
        raise AuthenticationError(detail="Invalid API key format")

    prefix = api_key.split('_', 1)[0]
    candidates = await db.get_api_keys_by_prefix(prefix)

    valid_data = None
    for kd in candidates:
        if key_manager.verify_api_key(api_key, kd.hashed_key):
            valid_data = kd
            break

    if not valid_data:
        raise AuthenticationError(detail="Invalid API key")

    if not valid_data.is_active or (valid_data.expires_at and valid_data.expires_at < datetime.now(timezone.utc)):
        raise AuthenticationError(detail="API key inactive or expired")

    user = await db.get_user(valid_data.user_id)
    if not user or user.disabled:
        raise AuthenticationError(detail="User not found or disabled")

    request.state.validated_api_key = valid_data
    asyncio.create_task(db.update_api_key_usage(valid_data.key_id))

    return user

# --- Bearer Token Scheme (HS256 or Google ID‑token RS256) ---
bearer_scheme = HTTPBearer(auto_error=False)

def verify_token(token: str) -> Dict[str, Any]:
    # 1) Try internal HS256
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            audience=settings.API_AUDIENCE,
        )
        logger.debug("Internal JWT validated")
        return payload
    except PyJWTError as e:
        logger.debug(f"HS256 decode failed ({e}), falling back to Google ID‑token")

    # 2) Fallback to Google ID‑token (RS256)
    try:
        req = google_requests.Request()
        payload = google_id_token.verify_oauth2_token(
            token, req, settings.BACKEND_URL
        )
        logger.debug("Google ID‑token validated")
        return payload
    except Exception as e:
        logger.warning(f"Google ID‑token validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def validate_bearer(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> User:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token missing")
    data = verify_token(credentials.credentials)
    user_id = data.get("sub") or data.get("user_id") or data.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = User.get_by_id_or_email(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=403, detail="Inactive or unknown user")
    return user

# --- Combined Dependency to allow either API‑key or Bearer ---
async def get_current_user(
    api_user: Optional[User] = Depends(validate_api_key),
    bearer_user: Optional[User] = Depends(validate_bearer),
) -> User:
    return api_user or bearer_user

# --- Security Headers Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        resp = await call_next(request)
        cfg = get_settings()
        if cfg.is_production:
            resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["X-XSS-Protection"] = "1; mode=block"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return resp

def setup_security(app):
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware applied.")