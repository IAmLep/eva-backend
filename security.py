"""
Security module for EVA backend.

Handles API key authentication, scope checking, and potentially other
security mechanisms like headers (CSP, HSTS).
"""

import logging
import secrets
import time
import asyncio # Import asyncio
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union, Callable # Import Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, APIKeyQuery, OAuth2PasswordBearer
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware # Import BaseHTTPMiddleware

# --- Local Imports ---
from config import get_settings
from database import get_db_manager, DatabaseManager
from exceptions import AuthenticationError, AuthorizationError
from models import User, ApiKey, ApiKeyScope

# --- Setup ---
logger = logging.getLogger(__name__)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)
api_key_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Security Configuration ---
class SecurityConfig:
    def __init__(self):
        self.settings = get_settings()
        self.hsts_enabled = self.settings.is_production
        self.hsts_max_age = 31536000

    def apply_security_headers(self, response: Response) -> None:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if self.hsts_enabled:
            response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains"

_security_config: Optional[SecurityConfig] = None
def get_security_config() -> SecurityConfig:
    global _security_config
    if _security_config is None: _security_config = SecurityConfig()
    return _security_config

# --- API Key Management Logic ---
class APIKeyManager:
    PREFIX_LENGTH = 8
    SECRET_LENGTH = 32

    def generate_api_key(self) -> Tuple[str, str, str]: # Added prefix return
        prefix = secrets.token_urlsafe(self.PREFIX_LENGTH)
        secret_part = secrets.token_urlsafe(self.SECRET_LENGTH)
        full_key = f"{prefix}_{secret_part}"
        hashed_key = self.hash_api_key(full_key)
        return full_key, hashed_key, prefix # Return prefix

    def hash_api_key(self, key: str) -> str:
        return api_key_context.hash(key)

    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        try: return api_key_context.verify(plain_key, hashed_key)
        except Exception: return False # Logged below

    def validate_api_key_format(self, key: str) -> bool:
        parts = key.split('_')
        return len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0

_api_key_manager: Optional[APIKeyManager] = None
def get_api_key_manager() -> APIKeyManager:
    global _api_key_manager
    if _api_key_manager is None: _api_key_manager = APIKeyManager()
    return _api_key_manager

# --- Authentication Dependencies ---
async def get_api_key(
    api_key_query: Annotated[Optional[str], Depends(API_KEY_QUERY)] = None,
    api_key_header: Annotated[Optional[str], Depends(API_KEY_HEADER)] = None,
) -> Optional[str]:
    return api_key_header or api_key_query

async def validate_api_key(
    request: Request, # Inject request to store state
    api_key: Annotated[Optional[str], Depends(get_api_key)] = None,
    db: DatabaseManager = Depends(get_db_manager),
    key_manager: APIKeyManager = Depends(get_api_key_manager)
) -> User:
    """
    Validates API key using prefix lookup and hash verification.
    Stores validated ApiKey object in request.state.
    """
    if not api_key: raise AuthenticationError(detail="API key required")
    if not key_manager.validate_api_key_format(api_key):
        raise AuthenticationError(detail="Invalid API key format")

    prefix = api_key.split('_')[0]
    potential_keys = await db.get_api_keys_by_prefix(prefix) # Requires DB implementation

    validated_key_data: Optional[ApiKey] = None
    for key_data in potential_keys:
        if key_manager.verify_api_key(api_key, key_data.hashed_key):
            validated_key_data = key_data
            break # Found matching key

    if not validated_key_data:
        logger.warning(f"API key validation failed: No key found matching prefix '{prefix}' and provided secret.")
        raise AuthenticationError(detail="Invalid API key") # Keep error generic

    # --- Key Found - Perform Checks ---
    if not validated_key_data.is_active:
        raise AuthenticationError(detail="API key is inactive")
    if validated_key_data.expires_at and validated_key_data.expires_at < datetime.now(timezone.utc):
        raise AuthenticationError(detail="API key has expired")

    # --- Fetch Associated User ---
    user = await db.get_user(validated_key_data.user_id)
    if not user: raise AuthenticationError(detail="User for API key not found")
    if user.disabled: raise AuthenticationError(detail="User account is disabled")

    # --- Store validated key in request state for scope checking ---
    request.state.validated_api_key = validated_key_data

    # --- Update Last Used Timestamp ---
    asyncio.create_task(db.update_api_key_usage(validated_key_data.key_id))

    logger.info(f"API key validated successfully for user {user.id} (Key ID: {validated_key_data.key_id})")
    return user

# --- Authorization Dependency ---
def require_api_scope(required_scope: Union[ApiKeyScope, str]):
    async def dependency(
        request: Request, # Inject request to access state
        current_user: Annotated[User, Depends(validate_api_key)] # Ensures API key is validated first
    ) -> None:
        """Checks if the validated API key (from request.state) has the required scope."""
        validated_key_data: Optional[ApiKey] = getattr(request.state, "validated_api_key", None)

        if not validated_key_data:
            logger.error("Could not find validated API key data in request state for scope check.")
            # This indicates an internal logic error if validate_api_key ran successfully
            raise AuthorizationError(detail="Internal error checking API key scope")

        try:
            required_scope_enum = ApiKeyScope(required_scope) if isinstance(required_scope, str) else required_scope
        except ValueError:
             raise AuthorizationError(detail=f"Invalid scope required: {required_scope}")

        # Check if the required scope (or admin scope) is present
        has_scope = required_scope_enum in validated_key_data.scopes or ApiKeyScope.ADMIN in validated_key_data.scopes

        if not has_scope:
            logger.warning(f"Authorization failed: User {current_user.id} (Key {validated_key_data.key_id}) missing scope '{required_scope_enum.value}'.")
            raise AuthorizationError(detail=f"Requires scope: {required_scope_enum.value}")
        else:
             logger.debug(f"Scope '{required_scope_enum.value}' authorized for user {current_user.id} (Key {validated_key_data.key_id}).")

    return dependency

# --- Security Middleware ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
     async def dispatch(self, request: Request, call_next: Callable) -> Response:
         response = await call_next(request)
         sec_config = get_security_config()
         sec_config.apply_security_headers(response)
         return response

# --- Setup Function ---
def setup_security(app: FastAPI) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware applied.")
