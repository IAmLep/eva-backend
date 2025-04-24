import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext

# Local imports
from config import settings
# Corrected: Import the manager getter, not the specific function
from database import get_db_manager
from models import TokenData, User, UserInDB # Keep UserInDB
from exceptions import DatabaseError, NotFoundException # Keep NotFoundException

logger = logging.getLogger(__name__)


# --- Security Setup ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token") # Points to the /token endpoint

# --- Helper Functions ---
def verify_password(plain_password, hashed_password):
    """Verifies a plain password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hashes a password using bcrypt."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

# --- Core Authentication Logic ---
async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticates a user by username and password."""
    # Corrected: Get the database manager instance
    db_manager = get_db_manager()
    try:
        # Corrected: Use the manager instance to call the method
        user = await db_manager.get_user_by_username(username)
        if not user:
            logger.warning(f"Authentication attempt failed: User '{username}' not found.")
            return None
        if not verify_password(password, user.hashed_password):
            logger.warning(f"Authentication attempt failed: Invalid password for user '{username}'.")
            return None
        # Check if user is active (if applicable)
        # if user.disabled:
        #     logger.warning(f"Authentication attempt failed: User '{username}' is disabled.")
        #     return None
        logger.info(f"User '{username}' authenticated successfully.")
        return user
    except DatabaseError as e:
        logger.error(f"Database error during authentication for user '{username}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error during authentication for user '{username}': {e}", exc_info=True)
        return None


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """Decodes JWT token and retrieves the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Corrected: Get the database manager instance
    db_manager = get_db_manager()
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.warning("Token validation failed: 'sub' (username) missing in payload.")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.warning(f"Token validation failed: JWTError - {e}")
        raise credentials_exception

    try:
        # Corrected: Use the manager instance to call the method
        user = await db_manager.get_user_by_username(token_data.username)
        if user is None:
            logger.warning(f"Token validation failed: User '{token_data.username}' from token not found in DB.")
            raise credentials_exception
        # Optional: Check if user is disabled
        # if user.disabled:
        #     logger.warning(f"Token validation failed: User '{token_data.username}' is disabled.")
        #     raise credentials_exception
        return user
    except DatabaseError as e:
        logger.error(f"Database error retrieving user '{token_data.username}' from token: {e}", exc_info=True)
        raise credentials_exception
    except Exception as e:
        logger.error(f"Unexpected error retrieving user '{token_data.username}' from token: {e}", exc_info=True)
        raise credentials_exception

# Dependency to get the currently active user (can add checks like user.disabled)
async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    # if current_user.disabled:
    #     raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- User Management Functions (moved from main.py or elsewhere if needed) ---
# Ensure any other database interactions here also use get_db_manager()