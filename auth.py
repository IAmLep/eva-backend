import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# --- Google Auth Imports ---
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests
# --- End Google Auth Imports ---

# Local imports
from config import settings
from database import get_db_manager
from models import TokenData, User, UserInDB
from exceptions import DatabaseError, NotFoundException

logger = logging.getLogger(__name__)

# --- Security Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# --- Helper Functions ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hashes a password using bcrypt."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

# --- Core Authentication Logic ---
async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user by username and password."""
    db_manager = get_db_manager()
    try:
        user = await db_manager.get_user_by_username(username)
        if not user or not verify_password(password, user.hashed_password):
            return None
        return user
    except DatabaseError as e:
        logger.error(f"Database error during authentication for user '{username}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during authentication for user '{username}': {e}", exc_info=True)
        return None

# --- get_current_user handles multiple auth methods ---
async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> UserInDB:
    """
    Authenticates a request by:
    1) Cloud Run header if present,
    2) Internal HS256 JWT,
    3) Google ID Token RS256.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    db_manager = get_db_manager()
    user: Optional[UserInDB] = None

    # 1) Cloud Run validated ID token header
    header_email = request.headers.get("X-Goog-Authenticated-User-Email")
    if header_email:
        email = header_email.split(":")[-1]
        try:
            user = await db_manager.get_user_by_email(email)
        except DatabaseError:
            raise credentials_exception
        if not user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Authenticated user '{email}' not registered.",
            )
        return user

    # 2) Bearer token provided
    if token:
        # 2a) Try internal HS256 JWT
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
                audience=settings.API_AUDIENCE,
            )
            username = payload.get("sub")
            if username:
                user = await db_manager.get_user_by_username(username)
                if user:
                    return user
        except JWTError:
            pass

        # 2b) Try Google ID token validation
        try:
            if not settings.BACKEND_URL:
                raise credentials_exception
            google_req = google_requests.Request()
            payload = google_id_token.verify_oauth2_token(
                token, google_req, audience=settings.BACKEND_URL
            )
            email = payload.get("email")
            if email:
                user = await db_manager.get_user_by_email(email)
                if user:
                    return user
        except Exception:
            pass

    # 3) No valid authentication
    raise credentials_exception

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Returns the authenticated user (can add disabled checks here)."""
    return current_user

# --- Placeholder for encryption key retrieval ---
async def get_user_encryption_key(user_id: str) -> bytes:
    """
    Placeholder function to retrieve user-specific encryption key.
    WARNING: Implement proper key management. This is NOT secure.
    """
    logger.warning("Using placeholder encryption key retrieval; implement secure KMS.")
    import hashlib
    material = f"{user_id}-{settings.SECRET_KEY}".encode()
    return hashlib.sha256(material).digest()

"""file end"""