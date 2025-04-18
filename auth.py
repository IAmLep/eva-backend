"""
Authentication Logic module for EVA backend.

Handles password hashing/verification, JWT token creation/decoding,
user authentication checks, and dependency functions for securing endpoints.
Does NOT contain API route definitions.
"""

import logging
from datetime import datetime, timedelta, timezone # Use timezone-aware datetimes
from typing import Annotated, Dict, Optional, Union

# --- FastAPI & Security ---
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# --- Google Auth ---
# Use try-except for robustness
try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False
    google_requests = None
    id_token = None

# --- Local Imports ---
from config import get_settings
# Import the specific DB function needed here
from database import get_user_by_username # Use the specific function
from exceptions import AuthenticationError, AuthorizationError
# Import models used for type hinting and data structures
from models import User, UserInDB # Use UserInDB for internal representation

# --- Setup ---

# Logger configuration
logger = logging.getLogger(__name__)

# Password hashing context using bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer scheme (points to the login endpoint)
# Note: tokenUrl should match the path defined in auth_router.py
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login") # Updated path

# --- Pydantic Models for Token Data ---

class TokenData(BaseModel):
    """Data extracted from JWT payload."""
    username: Optional[str] = None # Subject claim ('sub') usually holds username

# --- Password Utilities ---

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against its hashed version."""
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        # Log unexpected errors during verification but treat as mismatch
        logger.error(f"Error verifying password hash: {e}", exc_info=True)
        return False

def get_password_hash(password: str) -> str:
    """Generates a bcrypt hash for a given password."""
    return pwd_context.hash(password)

# --- User Authentication ---

async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """
    Authenticates a user based on username and password.

    Fetches the user from the database and verifies the password.

    Args:
        username: The username to authenticate.
        password: The password to verify.

    Returns:
        The UserInDB object if authentication is successful, otherwise None.
    """
    user = await get_user_by_username(username)
    if not user:
        logger.warning(f"Authentication failed: User '{username}' not found.")
        return None
    # Ensure we have the hashed password (UserInDB includes it)
    # If get_user_by_username returns User, we need UserInDB
    # Assuming get_user_by_username can return UserInDB or similar with hash
    if not hasattr(user, 'hashed_password') or not user.hashed_password:
         logger.error(f"Authentication error: Hashed password missing for user '{username}'.")
         return None # Cannot verify without hash

    if not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed: Invalid password for user '{username}'.")
        return None

    # Return the full UserInDB object on success
    # Ensure the object returned by get_user_by_username is compatible
    # If it returns User, we might need another DB call or adjust get_user_by_username
    # For now, assuming it returns an object convertible to UserInDB
    if isinstance(user, User) and not isinstance(user, UserInDB):
         # If we only got a User model, try fetching UserInDB explicitly (less efficient)
         # This depends heavily on database.py implementation details.
         # Ideally, get_user_by_username should fetch the necessary fields.
         logger.warning(f"authenticate_user received User model, expected UserInDB for {username}.")
         # Let's assume for now get_user_by_username returns required fields or is UserInDB
         pass # Proceed, assuming 'user' has hashed_password

    # Convert to UserInDB if necessary and possible, otherwise trust the input type
    try:
        user_in_db = UserInDB(**user.model_dump())
        return user_in_db
    except Exception as e:
         logger.error(f"Failed to cast user {username} to UserInDB: {e}")
         return None # Failed conversion


# --- JWT Token Utilities ---

def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT access token.

    Args:
        data: Data to encode in the token payload (e.g., {'sub': username}).
        expires_delta: Optional timedelta for token expiration. Defaults to config setting.

    Returns:
        The encoded JWT token string.
    """
    settings = get_settings()
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)}) # Add issued-at time

    try:
        encoded_jwt = jwt.encode(
            to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
        )
        return encoded_jwt
    except JWTError as e:
         logger.error(f"Failed to encode JWT: {e}", exc_info=True)
         raise AuthenticationError(detail="Could not create access token.")


async def decode_access_token(token: str) -> TokenData:
     """
     Decodes a JWT access token and returns the payload data.

     Args:
         token: The JWT token string.

     Returns:
         TokenData containing the username.

     Raises:
         AuthenticationError: If the token is invalid, expired, or missing data.
     """
     settings = get_settings()
     credentials_exception = AuthenticationError(
         detail="Could not validate credentials",
         headers={"WWW-Authenticate": "Bearer"},
     )

     try:
         payload = jwt.decode(
             token,
             settings.SECRET_KEY,
             algorithms=[settings.ALGORITHM],
             options={"verify_aud": False} # Add audience verification if needed
         )
         username: Optional[str] = payload.get("sub")
         if username is None:
             logger.error("Token validation failed: 'sub' claim (username) missing.")
             raise credentials_exception

         # Check expiration (already handled by jwt.decode, but good practice)
         # exp = payload.get("exp")
         # if exp is None or datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
         #     logger.warning("Token validation failed: Token expired.")
         #     raise AuthenticationError(detail="Token has expired", headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""})

         token_data = TokenData(username=username)
         return token_data

     except JWTError as e:
         logger.warning(f"Token validation failed: {e}")
         if "expired" in str(e).lower():
              raise AuthenticationError(detail="Token has expired", headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""})
         else:
              raise credentials_exception # General validation error


# --- FastAPI Dependency Functions ---

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """
    FastAPI dependency to get the current user from a JWT token.

    Verifies the token and fetches the corresponding user from the database.

    Args:
        token: The JWT token extracted from the request by OAuth2PasswordBearer.

    Returns:
        The authenticated User object.

    Raises:
        AuthenticationError: If authentication fails (invalid token, user not found).
    """
    if token is None:
         # This case might happen if auto_error=False in OAuth2PasswordBearer
         # and no token is provided. Handle it explicitly.
         raise AuthenticationError(
              detail="Not authenticated",
              headers={"WWW-Authenticate": "Bearer"}
         )

    token_data = await decode_access_token(token) # Handles decoding and validation exceptions

    user = await get_user_by_username(token_data.username)
    if user is None:
        logger.error(f"Token validation failed: User '{token_data.username}' not found in database.")
        # Raise specific exception even if token was valid, user doesn't exist anymore
        raise AuthenticationError(
            detail="User associated with token not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Return the User model (without sensitive data)
    return User(**user.model_dump())


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    FastAPI dependency ensuring the authenticated user is active (not disabled).

    Args:
        current_user: The user object obtained from get_current_user dependency.

    Returns:
        The active User object.

    Raises:
        AuthorizationError: If the user is disabled.
    """
    # We fetch the user again inside get_current_user, so the disabled status should be current.
    # If User model didn't include 'disabled', we'd fetch again here.
    if current_user.disabled:
        logger.warning(f"Access attempt by disabled user: {current_user.username} (ID: {current_user.id})")
        raise AuthorizationError(detail="Inactive user")
    return current_user


# --- Google ID Token Validation ---

async def validate_google_id_token(token: str) -> Dict[str, Any]:
    """
    Validates a Google-issued ID token.

    Args:
        token: The Google ID token string.

    Returns:
        The token payload dictionary if validation is successful.

    Raises:
        AuthenticationError: If the token is invalid or expired, or auth library unavailable.
        ConfigurationError: If required settings (SERVICE_URL) are missing.
    """
    if not GOOGLE_AUTH_AVAILABLE:
        logger.error("Google Auth library is not available. Cannot validate Google ID token.")
        raise AuthenticationError(detail="Google Sign-In not supported by server configuration.")

    settings = get_settings()
    # Google requires an audience check. Use SERVICE_URL or fallback to a configured client ID.
    # Using SERVICE_URL is typical for Cloud Run.
    audience = str(settings.SERVICE_URL) if settings.SERVICE_URL else None
    if not audience:
        # Alternatively, use a Client ID configured in settings if SERVICE_URL isn't reliable
        # audience = settings.GOOGLE_CLIENT_ID
        logger.error("SERVICE_URL is not configured. Cannot validate Google ID token audience.")
        raise ConfigurationError("Server configuration missing required URL for Google token validation.")

    try:
        request = google_requests.Request()
        # Verify the token against Google's public keys and check audience/issuer.
        payload = id_token.verify_oauth2_token(
            token,
            request,
            audience=audience
            # Can also specify clock_skew_in_seconds if needed
        )

        # Optional: Check issuer if necessary
        # if payload['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
        #     raise AuthenticationError(detail="Invalid token issuer.")

        # id_token.verify_oauth2_token already checks expiration.
        # Manual check is redundant unless custom logic needed.
        # if datetime.fromtimestamp(payload['exp'], timezone.utc) < datetime.now(timezone.utc):
        #     logger.error("Google ID token has expired (should have been caught by verify_oauth2_token).")
        #     raise AuthenticationError(detail="Token expired")

        logger.info(f"Google ID token validated successfully for email: {payload.get('email')}")
        return payload

    except ValueError as e:
        # Catches errors from verify_oauth2_token like invalid format, signature, expired, audience mismatch etc.
        logger.warning(f"Google ID token validation failed: {e}")
        raise AuthenticationError(
            detail=f"Invalid Google ID token: {e}",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""}
        )
    except Exception as e:
        # Catch other potential errors (network issues during key fetching, etc.)
        logger.error(f"Unexpected error validating Google ID token: {e}", exc_info=True)
        raise AuthenticationError(
            detail=f"Could not validate Google ID token: {e}",
            headers={"WWW-Authenticate": "Bearer error=\"invalid_token\""}
        )