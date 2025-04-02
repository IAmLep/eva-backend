"""
Authentication module for EVA backend.

This module handles user authentication, JWT token generation, and validation.
It provides functions for securing API endpoints and managing user sessions.

Last updated: 2025-04-02
Version: v1.8.7
"""

import logging
from datetime import datetime, timedelta
from typing import Annotated, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from config import get_settings
from database import get_user_by_username, verify_user_exists
from exceptions import AuthenticationError, AuthorizationError
from models import User, UserInDB

# Setup router
router = APIRouter()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 password bearer scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# Logger configuration
logger = logging.getLogger(__name__)


class Token(BaseModel):
    """
    Token response model.
    
    Attributes:
        access_token: JWT access token
        token_type: Type of token (always 'bearer')
        expires_at: Timestamp when token expires
    """
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    """
    Token data extracted from JWT.
    
    Attributes:
        username: Username extracted from token
        scopes: Optional list of permission scopes
    """
    username: str
    scopes: Optional[list[str]] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against a hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password to compare against
        
    Returns:
        bool: True if password matches hash, False otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    
    Args:
        password: Plain text password to hash
        
    Returns:
        str: Hashed password
    """
    return pwd_context.hash(password)


async def authenticate_user(username: str, password: str) -> Union[UserInDB, None]:
    """
    Authenticate a user with username and password.
    
    Args:
        username: Username to authenticate
        password: Password to verify
        
    Returns:
        UserInDB: User object if authentication is successful
        None: If authentication fails
    """
    user = await get_user_by_username(username)
    if not user:
        logger.warning(f"Authentication failed: User {username} not found")
        return None
    if not verify_password(password, user.hashed_password):
        logger.warning(f"Authentication failed: Invalid password for user {username}")
        return None
    return user


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        str: Encoded JWT token
    """
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    
    return encoded_jwt


async def validate_google_id_token(token: str) -> dict:
    """
    Validate a Google ID token.
    
    Args:
        token: Google-issued ID token
        
    Returns:
        dict: Token payload if valid
        
    Raises:
        AuthenticationError: If token is invalid
    """
    try:
        # Get the expected audience (your service URL)
        settings = get_settings()
        audience = settings.SERVICE_URL or "https://eva-backend-533306620971.europe-west1.run.app"
        
        # Verify the token
        request = google_requests.Request()
        payload = id_token.verify_oauth2_token(
            token,
            request,
            audience=audience
        )
        
        # Check if token is expired
        if datetime.fromtimestamp(payload['exp']) < datetime.utcnow():
            logger.error("Google ID token has expired")
            raise AuthenticationError(detail="Token expired")
            
        return payload
    except Exception as e:
        logger.error(f"Google ID token validation failed: {str(e)}")
        raise AuthenticationError(
            detail=f"Invalid Google ID token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """
    Get the current user from JWT token.
    
    Args:
        token: JWT token from authorization header
        
    Returns:
        User: Current authenticated user
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    settings = get_settings()
    credentials_exception = AuthenticationError(
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.error("Token validation failed: missing subject claim")
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise credentials_exception
    
    user = await get_user_by_username(token_data.username)
    if user is None:
        logger.error(f"Token validation failed: user {token_data.username} not found")
        raise credentials_exception
    
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Get the current active user.
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        User: Current active user
        
    Raises:
        AuthorizationError: If user is disabled
    """
    if current_user.disabled:
        logger.warning(f"Access attempt by disabled user: {current_user.username}")
        raise AuthorizationError(detail="Inactive user")
    return current_user


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Dict:
    """
    Login endpoint to get access token.
    
    Args:
        form_data: OAuth2 form with username and password
        
    Returns:
        Dict: Token response with access_token, token_type, and expires_at
        
    Raises:
        HTTPException: If authentication fails
    """
    settings = get_settings()
    user = await authenticate_user(form_data.username, form_data.password)
    
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Calculate expiration time
    expires_delta = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.utcnow() + expires_delta
    
    # Create token with subject as username
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=expires_delta
    )
    
    logger.info(f"Generated access token for user: {user.username}")
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "expires_at": expires_at
    }


async def validate_request_auth(request: Request) -> User:
    """
    Validate authentication from request.
    
    Args:
        request: FastAPI request object
        
    Returns:
        User: Authenticated user
        
    Raises:
        AuthenticationError: If authentication is invalid
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        logger.warning("Missing Authorization header in request")
        raise AuthenticationError(detail="Missing Authorization header")
    
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning(f"Invalid Authorization header format: {auth_header}")
        raise AuthenticationError(detail="Invalid Authorization header format")
    
    token = parts[1]
    
    # First try to validate as Google ID token for service accounts
    try:
        payload = await validate_google_id_token(token)
        # For service account tokens, create or get user
        email = payload.get('email', '')
        if email.endswith('.gserviceaccount.com'):
            # Return a system user for service accounts
            logger.info(f"Authenticated service account: {email}")
            return User(
                username=email,
                email=email,
                full_name="Service Account",
                disabled=False,
                is_service_account=True,
                scopes=["api:all"]  # Grant full access to service accounts
            )
    except AuthenticationError as e:
        # If Google token validation fails, try regular JWT token
        logger.debug(f"Google ID token validation failed, trying JWT: {str(e)}")
        pass
    
    # Fall back to regular JWT validation
    try:
        return await get_current_user(token)
    except Exception as e:
        logger.error(f"Authentication failed after trying both methods: {str(e)}")
        raise AuthenticationError(detail="Invalid authentication token")