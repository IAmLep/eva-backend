"""
Security module for EVA backend.

Handles API key authentication, scope checking, and potentially other
security mechanisms like headers (CSP, HSTS).
CSRF logic is removed as it's less relevant for a pure API/WebSocket backend.
"""

import logging
import secrets # Use secrets module for secure random generation
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, APIKeyQuery, OAuth2PasswordBearer
from passlib.context import CryptContext # For hashing API keys securely

# --- Local Imports ---
from config import get_settings
# Assuming database.py provides these methods now
from database import get_db_manager, DatabaseManager
from exceptions import AuthenticationError, AuthorizationError
# Import models used
from models import User, ApiKey, ApiKeyScope # Import ApiKey model

# --- Setup ---
logger = logging.getLogger(__name__)

# API Key Retrieval Mechanisms (Header and Query Parameter)
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
API_KEY_QUERY = APIKeyQuery(name="api_key", auto_error=False)

# Password context specifically for hashing API keys (separate from user passwords)
# Use a strong hashing algorithm like Argon2 if available and suitable, otherwise bcrypt
# For simplicity, using bcrypt here, but Argon2 is generally recommended for API keys if possible.
api_key_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Security Configuration ---
class SecurityConfig:
    """Holds security-related configurations and provides helper methods."""

    def __init__(self):
        self.settings = get_settings()
        # Example: Strict-Transport-Security (HSTS) - Enable in production via proxy recommended
        self.hsts_enabled = self.settings.is_production
        self.hsts_max_age = 31536000 # 1 year

    def apply_security_headers(self, response: Response) -> None:
        """Applies common security headers to responses."""
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        # Enable browser XSS filtering (mostly superseded by CSP, but good fallback)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (CSP) - Define a suitable policy
        # This is a VERY restrictive example, adjust based on actual needs (scripts, styles, etc.)
        # csp_policy = "default-src 'self'; object-src 'none'; frame-ancestors 'none'; upgrade-insecure-requests;"
        # response.headers["Content-Security-Policy"] = csp_policy

        # HSTS - Handle carefully, best applied at proxy/load balancer level
        if self.hsts_enabled:
            response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains"

    # Input sanitization placeholder (basic example)
    def sanitize_input(self, text: str) -> str:
        """Basic input sanitization (replace with a proper library like bleach if handling HTML)."""
        if not isinstance(text, str): return text # Avoid errors on non-strings
        # Simple example: escape basic HTML chars. NOT sufficient for preventing XSS if rendering HTML.
        # For API data, validation is usually more important than sanitization.
        sanitized = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if sanitized != text:
             logger.debug("Input sanitized (basic HTML escape).")
        return sanitized

# --- Singleton Instance ---
_security_config: Optional[SecurityConfig] = None

def get_security_config() -> SecurityConfig:
    """Gets the singleton SecurityConfig instance."""
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig()
    return _security_config

# --- API Key Management Logic ---
class APIKeyManager:
    """Handles generation, hashing, and validation of API keys."""

    PREFIX_LENGTH = 8
    SECRET_LENGTH = 32 # Length of the random part of the key

    def generate_api_key(self) -> Tuple[str, str]:
        """Generates a new API key (prefix + secret) and its hash."""
        prefix = secrets.token_urlsafe(self.PREFIX_LENGTH)
        secret_part = secrets.token_urlsafe(self.SECRET_LENGTH)
        full_key = f"{prefix}_{secret_part}"
        hashed_key = self.hash_api_key(full_key)
        return full_key, hashed_key, prefix

    def hash_api_key(self, key: str) -> str:
        """Hashes an API key using the configured context."""
        return api_key_context.hash(key)

    def verify_api_key(self, plain_key: str, hashed_key: str) -> bool:
        """Verifies a plain API key against a stored hash."""
        try:
            return api_key_context.verify(plain_key, hashed_key)
        except Exception as e:
            logger.error(f"Error verifying API key hash: {e}", exc_info=True)
            return False

    def validate_api_key_format(self, key: str) -> bool:
        """Checks if the key format seems valid (prefix_secret)."""
        parts = key.split('_')
        # Basic check for prefix and secret parts
        return len(parts) == 2 and len(parts[0]) > 0 and len(parts[1]) > 0

# --- Singleton Instance ---
_api_key_manager: Optional[APIKeyManager] = None

def get_api_key_manager() -> APIKeyManager:
    """Gets the singleton APIKeyManager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


# --- Authentication Dependencies ---

async def get_api_key(
    api_key_query: Annotated[Optional[str], Depends(API_KEY_QUERY)] = None,
    api_key_header: Annotated[Optional[str], Depends(API_KEY_HEADER)] = None,
) -> Optional[str]:
    """
    FastAPI dependency to extract API key from query parameter or header.
    Prioritizes header if both are present.
    """
    if api_key_header:
        return api_key_header
    if api_key_query:
        return api_key_query
    return None # No key provided

async def validate_api_key(
    api_key: Annotated[Optional[str], Depends(get_api_key)] = None,
    db: DatabaseManager = Depends(get_db_manager),
    key_manager: APIKeyManager = Depends(get_api_key_manager)
) -> User:
    """
    FastAPI dependency to validate the provided API key and return the associated user.

    - Checks if a key is provided.
    - Validates the key format (basic check).
    - Hashes the provided key.
    - Looks up the hashed key in the database.
    - Checks if the key is active and not expired.
    - Fetches the associated user.
    - Updates the key's last_used timestamp.

    Raises:
        AuthenticationError: If the key is missing, invalid, inactive, expired,
                             or the associated user is not found or inactive.
    """
    if not api_key:
        logger.debug("API key validation failed: No key provided.")
        raise AuthenticationError(detail="API key required")

    # Basic format validation before hashing
    if not key_manager.validate_api_key_format(api_key):
        logger.warning("API key validation failed: Invalid format.")
        raise AuthenticationError(detail="Invalid API key format")

    # Hash the provided key to look it up in the DB
    # Note: This means we cannot verify the key directly, only find it by hash.
    # This is less secure than verifying. A better approach stores only the hash
    # and requires the client to send the full key for verification against the hash.
    # Let's adapt to verify against stored hash.
    # We need to retrieve potential keys based on prefix first, then verify.

    # Alternative (Better) Approach: Retrieve by prefix, then verify full key
    prefix = api_key.split('_')[0]
    # TODO: Need a DB method `get_api_keys_by_prefix(prefix)`
    # potential_keys = await db.get_api_keys_by_prefix(prefix)
    # For now, sticking to the less secure hash lookup shown in original files:
    hashed_provided_key = key_manager.hash_api_key(api_key) # Hash the key client sent

    # Find key by this hash in DB (assuming only hash is stored, which isn't ideal)
    # Let's assume `get_api_key_by_hash` looks up based on a stored HASH, not the key itself.
    api_key_data: Optional[ApiKey] = await db.get_api_key_by_hash(hashed_provided_key)

    if not api_key_data:
        logger.warning(f"API key validation failed: Key hash not found.")
        # Avoid indicating whether key exists for security (timing attacks)
        raise AuthenticationError(detail="Invalid API key") # Generic error

    # --- Key Found - Perform Checks ---
    if not api_key_data.is_active:
        logger.warning(f"API key validation failed: Key {api_key_data.key_id} is inactive.")
        raise AuthenticationError(detail="API key is inactive")

    if api_key_data.expires_at and api_key_data.expires_at < datetime.now(timezone.utc):
        logger.warning(f"API key validation failed: Key {api_key_data.key_id} has expired.")
        raise AuthenticationError(detail="API key has expired")

    # --- Fetch Associated User ---
    user = await db.get_user(api_key_data.user_id)
    if not user:
        logger.error(f"API key validation failed: User {api_key_data.user_id} associated with key {api_key_data.key_id} not found.")
        # Key is valid but user doesn't exist - treat as auth error
        raise AuthenticationError(detail="User for API key not found")

    if user.disabled:
        logger.warning(f"API key validation failed: User {user.id} associated with key {api_key_data.key_id} is disabled.")
        raise AuthenticationError(detail="User account is disabled")

    # --- Update Last Used Timestamp (fire-and-forget) ---
    asyncio.create_task(db.update_api_key_usage(api_key_data.key_id))

    logger.info(f"API key validated successfully for user {user.id} (Key ID: {api_key_data.key_id})")
    # Return the User model associated with the key
    return user

# --- Authorization Dependency ---
def require_api_scope(required_scope: Union[ApiKeyScope, str]):
    """
    FastAPI dependency factory to require a specific scope for an API key.
    """
    async def dependency(
        # Use the API key validation dependency to get the user and implicitly the key used
        request: Request, # Inject request to potentially access key scopes stored in state
        current_user: Annotated[User, Depends(validate_api_key)] # Depends on successful API key validation
    ) -> None:
        """Checks if the validated API key has the required scope."""
        # Retrieve the ApiKey object used for validation.
        # This requires validate_api_key to store it in request.state, or re-fetching it.
        # Let's assume we need to refetch based on user_id (less efficient).
        # TODO: Optimize by storing validated ApiKey in request.state in validate_api_key.

        # Fetching keys for user (inefficient, needs optimization)
        # db = get_db_manager()
        # user_keys = await db.get_user_api_keys(current_user.id) # Assumes this DB method exists
        # validated_key_data = next((k for k in user_keys if k.key_id == ???), None) # How to get key_id used?

        # --- Simplified approach: Assume validate_api_key stores scopes or we trust User model ---
        # This is insecure if User model doesn't reflect key scopes.
        # For now, let's raise NotImplementedError until key passing is refined.
        logger.error("Scope checking requires passing validated ApiKey data via request.state. Not implemented.")
        raise NotImplementedError("API scope checking mechanism needs refinement.")

        # --- Ideal Implementation (if ApiKey stored in request.state) ---
        # validated_key_data: Optional[ApiKey] = getattr(request.state, "validated_api_key", None)
        # if not validated_key_data:
        #     logger.error("Could not find validated API key data in request state for scope check.")
        #     raise AuthorizationError(detail="Internal error checking API key scope")
        #
        # required_scope_enum = ApiKeyScope(required_scope) if isinstance(required_scope, str) else required_scope
        #
        # # Check if the required scope (or admin scope) is present
        # has_scope = required_scope_enum in validated_key_data.scopes or ApiKeyScope.ADMIN in validated_key_data.scopes
        #
        # if not has_scope:
        #     logger.warning(f"Authorization failed: User {current_user.id} (Key {validated_key_data.key_id}) missing scope '{required_scope_enum.value}'.")
        #     raise AuthorizationError(detail=f"Requires scope: {required_scope_enum.value}")
        # else:
        #      logger.debug(f"Scope '{required_scope_enum.value}' authorized for user {current_user.id} (Key {validated_key_data.key_id}).")


    return dependency


# --- Security Middleware (Example) ---
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
     """Applies security headers defined in SecurityConfig."""
     async def dispatch(self, request: Request, call_next: Callable) -> Response:
         response = await call_next(request)
         sec_config = get_security_config()
         sec_config.apply_security_headers(response)
         return response

# --- Setup Function ---
def setup_security(app: FastAPI) -> None:
    """Applies security middleware to the FastAPI app."""
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware applied.")
    # Add other security middleware if needed (e.g., rate limiting middleware if not using dependencies)