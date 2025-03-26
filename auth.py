"""
Authentication utilities for the Eva LLM Application.
JWT token generation, validation, and related models.
"""
from datetime import datetime, timedelta
from typing import Optional, Union, Dict, Any, Tuple, List, Set
import uuid

from fastapi import Depends, HTTPException, status, Request, Form, Header
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
import base64
import secrets

from models import User, UserCreate, UserResponse
from database import get_db
from sqlalchemy.orm import Session
import os
from config import settings

# Secret key for JWT signing - should match your config.py SECRET_KEY
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS

# OAuth2 Client Credentials - for personal use
OAUTH2_CLIENT_ID = settings.OAUTH2_CLIENT_ID
OAUTH2_CLIENT_SECRET = settings.OAUTH2_CLIENT_SECRET
OAUTH2_SCOPES = settings.OAUTH2_SCOPES

# Token revocation storage
# In a production app, this would be persisted to a database
revoked_tokens = set()  # Set of JTIs (JWT IDs) that have been revoked
refresh_token_jti_map = {}  # Maps refresh token JTIs to their associated access token JTIs

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token endpoint
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes=OAUTH2_SCOPES,
)

class Token(BaseModel):
    """Schema for access tokens."""
    access_token: str
    token_type: str = "bearer"
    expires_at: Optional[datetime] = None
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    
    class Config:
        schema_extra = {"example": {"access_token": "eyJhbGciOiJ...", "token_type": "bearer", "expires_at": "2025-03-26T15:26:27Z"}}

class TokenData(BaseModel):
    """Data extracted from a token."""
    username: Optional[str] = None
    scopes: List[str] = []

class OAuth2ClientCredentials(BaseModel):
    """OAuth2 client credentials."""
    client_id: str
    client_secret: str
    
class OAuth2TokenRequestForm:
    """Form for OAuth2 token requests."""
    def __init__(
        self,
        *,
        grant_type: str = Form(None),
        username: str = Form(None),
        password: str = Form(None),
        scope: str = Form(""),
        client_id: Optional[str] = Form(None),
        client_secret: Optional[str] = Form(None),
    ):
        self.grant_type = grant_type
        self.username = username
        self.password = password
        self.scopes = scope.split()
        self.client_id = client_id
        self.client_secret = client_secret

class OAuth2TokenRefreshForm:
    """Form for OAuth2 token refresh requests."""
    def __init__(
        self,
        *,
        grant_type: str = Form(...),
        refresh_token: str = Form(...),
        client_id: Optional[str] = Form(None),
        client_secret: Optional[str] = Form(None),
    ):
        self.grant_type = grant_type
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.client_secret = client_secret

# Authentication functions
def verify_password(plain_password, hashed_password):
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generate a password hash."""
    return pwd_context.hash(password)

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify a JWT token and return its payload.
    
    Args:
        token: JWT token
        
    Returns:
        Token payload if valid, None otherwise
    """
    try:
        # First remove the "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
            
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check if token has been revoked
        jti = payload.get("jti")
        if jti and jti in revoked_tokens:
            return None
            
        return payload
    except JWTError:
        return None

def validate_client(client_id: str, client_secret: str) -> bool:
    """
    Validate OAuth2 client credentials.
    For personal use - this does a simple check against configured credentials.
    
    Args:
        client_id: The client ID to validate
        client_secret: The client secret to validate
        
    Returns:
        True if valid, False otherwise
    """
    return client_id == OAUTH2_CLIENT_ID and client_secret == OAUTH2_CLIENT_SECRET

async def validate_client_from_header(request: Request) -> bool:
    """
    Validate client credentials from Authorization header.
    
    Args:
        request: FastAPI request object
        
    Returns:
        True if valid, False otherwise
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        return False
    
    try:
        # Extract and decode credentials
        encoded_credentials = auth_header[6:]  # Remove "Basic "
        decoded = base64.b64decode(encoded_credentials).decode("utf-8")
        client_id, client_secret = decoded.split(":", 1)
        return validate_client(client_id, client_secret)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None, scopes: List[str] = None):
    """Create a new JWT access token."""
    to_encode = data.copy()
    
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Add token-specific claims
    jti = str(uuid.uuid4())
    to_encode.update({
        "exp": expire,
        "jti": jti,
        "token_type": "access",
        "iat": datetime.utcnow(),
        "scopes": scopes or []
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt, expire, jti

def create_refresh_token(data: dict, scopes: List[str] = None):
    """Create a new JWT refresh token with longer expiration."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Add token-specific claims
    jti = str(uuid.uuid4())
    to_encode.update({
        "exp": expire,
        "jti": jti,
        "token_type": "refresh",
        "iat": datetime.utcnow(),
        "scopes": scopes or []
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt, jti

def create_tokens(data: dict, scopes: List[str] = None) -> Dict[str, Any]:
    """
    Create both access and refresh tokens in one call.
    
    Args:
        data: Dictionary containing token data (usually user info)
        scopes: List of OAuth2 scopes to include in the tokens
        
    Returns:
        Dictionary with tokens and expiration information
    """
    access_token, expire, access_jti = create_access_token(data, scopes=scopes)
    refresh_token, refresh_jti = create_refresh_token(data, scopes=scopes)
    
    # Store the relationship between refresh and access tokens
    refresh_token_jti_map[refresh_jti] = access_jti
    
    # Return token response
    expires_in = int((expire - datetime.utcnow()).total_seconds())
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": expires_in,
        "expires_at": expire,
        "refresh_token": refresh_token
    }

def get_client_credentials(
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract client credentials from either form data or Authorization header.
    
    Args:
        client_id: Client ID from form data
        client_secret: Client secret from form data
        authorization: Authorization header
        
    Returns:
        Tuple of (client_id, client_secret) or (None, None) if not found
    """
    if client_id and client_secret:
        return client_id, client_secret
    
    if authorization and authorization.startswith("Basic "):
        try:
            encoded = authorization[6:]  # Remove "Basic "
            decoded = base64.b64decode(encoded).decode("utf-8")
            client_id, client_secret = decoded.split(":", 1)
            return client_id, client_secret
        except Exception:
            pass
    
    return None, None

async def validate_token_request(
    form_data: OAuth2TokenRequestForm = Depends(),
    authorization: Optional[str] = Header(None)
) -> Tuple[str, str, List[str]]:
    """
    Validate token request, including client credentials and grant type.
    
    Args:
        form_data: OAuth2 token request form data
        authorization: Authorization header for client credentials
        
    Returns:
        Tuple of (username, password, scopes)
        
    Raises:
        HTTPException: If validation fails
    """
    client_id, client_secret = get_client_credentials(
        form_data.client_id, form_data.client_secret, authorization
    )
    
    if not client_id or not client_secret or not validate_client(client_id, client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if form_data.grant_type != "password":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant type: {form_data.grant_type}",
        )
    
    return form_data.username, form_data.password, form_data.scopes

async def validate_token_refresh(
    form_data: OAuth2TokenRefreshForm = Depends(),
    authorization: Optional[str] = Header(None)
) -> str:
    """
    Validate token refresh request, including client credentials and grant type.
    
    Args:
        form_data: OAuth2 token refresh form data
        authorization: Authorization header for client credentials
        
    Returns:
        Refresh token
        
    Raises:
        HTTPException: If validation fails
    """
    client_id, client_secret = get_client_credentials(
        form_data.client_id, form_data.client_secret, authorization
    )
    
    if not client_id or not client_secret or not validate_client(client_id, client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    if form_data.grant_type != "refresh_token":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant type: {form_data.grant_type}",
        )
    
    return form_data.refresh_token

async def get_current_user(
    security_scopes: SecurityScopes = SecurityScopes(),
    token: str = Depends(oauth2_scheme), 
    db: Session = Depends(get_db)
):
    """
    Validate token and return current user.
    
    Args:
        security_scopes: Security scopes
        token: JWT token
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid
    """
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
            
        # Check if token has been revoked
        jti = payload.get("jti")
        if jti and jti in revoked_tokens:
            raise credentials_exception
            
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(username=username, scopes=token_scopes)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
        
    # Validate scopes
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Not enough permissions. Required scope: {scope}",
                headers={"WWW-Authenticate": authenticate_value},
            )
    
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if user is active."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authenticate a user by username and password."""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def verify_refresh_token(refresh_token: str) -> Optional[str]:
    """
    Verify a refresh token and extract the username.
    
    Args:
        refresh_token: JWT refresh token
        
    Returns:
        Username if valid, None otherwise
    """
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        is_refresh = payload.get("token_type") == "refresh"
        jti = payload.get("jti")
        
        if not username or not is_refresh:
            return None
            
        # Check if token has been revoked
        if jti and jti in revoked_tokens:
            return None
            
        return username
    except JWTError:
        return None

def create_user(db: Session, user: UserCreate) -> User:
    """
    Create a new user in the database.
    
    Args:
        db: Database session
        user: User creation data
        
    Returns:
        Created user
    """
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get a user by username."""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get a user by email."""
    return db.query(User).filter(User.email == email).first()

def generate_password_reset_token(email: str) -> str:
    """Generate a password reset token."""
    expire = datetime.utcnow() + timedelta(hours=1)
    data = {"sub": email, "exp": expire, "type": "password_reset"}
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify a password reset token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "password_reset":
            return None
        return payload.get("sub")
    except JWTError:
        return None