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
import redis

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
DEVICE_TOKEN_EXPIRE_DAYS = 365  # 1 year for device tokens

# Device authentication
INITIAL_DEVICE_SECRET = os.environ.get("EVA_INITIAL_DEVICE_SECRET", "")
if not INITIAL_DEVICE_SECRET:
    # Generate a default secret for development (not for production)
    INITIAL_DEVICE_SECRET = secrets.token_urlsafe(32)
    print(f"WARNING: Using generated initial device secret: {INITIAL_DEVICE_SECRET}")

# OAuth2 Client Credentials - for personal use
OAUTH2_CLIENT_ID = settings.OAUTH2_CLIENT_ID
OAUTH2_CLIENT_SECRET = settings.OAUTH2_CLIENT_SECRET
OAUTH2_SCOPES = {"chat:read": "Read chat messages", "chat:write": "Send chat messages"}

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

# Redis connection for token blacklist
try:
    redis_client = redis.Redis(
        host=settings.REDIS_HOST, 
        port=settings.REDIS_PORT, 
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD,
        decode_responses=True
    )
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None

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
    
class DeviceToken(BaseModel):
    """Schema for device tokens."""
    device_token: str
    token_type: str = "device"
    expires_at: datetime
    expires_in: int
    device_id: str
    
    class Config:
        schema_extra = {"example": {"device_token": "eyJhbGciOiJ...", "token_type": "device", "expires_at": "2025-03-26T15:26:27Z"}}

class DeviceAuthRequest(BaseModel):
    """Request for device authentication."""
    device_id: str
    device_name: str
    device_model: Optional[str] = None
    initial_secret: str
    
    class Config:
        schema_extra = {"example": {"device_id": "device-123", "device_name": "My Phone", "device_model": "Pixel 6", "initial_secret": "your-secret-here"}}

class DeviceValidationResponse(BaseModel):
    """Response for device token validation."""
    valid: bool
    expires_at: Optional[datetime] = None
    expires_in: Optional[int] = None
    device_id: Optional[str] = None
    
    class Config:
        schema_extra = {"example": {"valid": True, "expires_at": "2025-03-26T15:26:27Z", "expires_in": 31536000, "device_id": "device-123"}}

# Authentication functions
def verify_password(plain_password, hashed_password):
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generate a password hash."""
    return pwd_context.hash(password)

def is_token_blacklisted(token_jti: str) -> bool:
    """
    Check if a token is blacklisted.
    
    Args:
        token_jti: JWT ID to check
        
    Returns:
        True if blacklisted, False otherwise
    """
    if redis_client:
        # Check Redis for blacklisted token
        return bool(redis_client.get(f"blacklist:{token_jti}"))
    else:
        # Fall back to in-memory set if Redis is unavailable
        return token_jti in revoked_tokens

def blacklist_token(token_jti: str, expires_in: int = None):
    """
    Add a token to the blacklist.
    
    Args:
        token_jti: JWT ID to blacklist
        expires_in: Seconds until token expires naturally (for Redis TTL)
    """
    if redis_client:
        # Add to Redis with TTL if provided
        if expires_in:
            redis_client.setex(f"blacklist:{token_jti}", expires_in, "1")
        else:
            redis_client.set(f"blacklist:{token_jti}", "1")
    else:
        # Fall back to in-memory set
        revoked_tokens.add(token_jti)

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
            
        # Decode the token with explicit algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check if token has been revoked
        jti = payload.get("jti")
        if jti and is_token_blacklisted(jti):
            return None
            
        return payload
    except JWTError:
        return None

def validate_device_token(token: str) -> DeviceValidationResponse:
    """
    Validate a device token and return its status.
    
    Args:
        token: Device JWT token
        
    Returns:
        DeviceValidationResponse with validity information
    """
    try:
        # Decode the token with explicit algorithm
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check if token is a device token
        token_type = payload.get("token_type")
        if token_type != "device":
            return DeviceValidationResponse(valid=False)
        
        # Check if token has been revoked
        jti = payload.get("jti")
        if jti and is_token_blacklisted(jti):
            return DeviceValidationResponse(valid=False)
        
        # Check if token has expired
        exp = payload.get("exp")
        if not exp:
            return DeviceValidationResponse(valid=False)
        
        exp_datetime = datetime.fromtimestamp(exp)
        now = datetime.utcnow()
        
        if exp_datetime <= now:
            return DeviceValidationResponse(valid=False)
        
        # Token is valid
        expires_in = int((exp_datetime - now).total_seconds())
        
        return DeviceValidationResponse(
            valid=True,
            expires_at=exp_datetime,
            expires_in=expires_in,
            device_id=payload.get("sub")
        )
    except JWTError:
        return DeviceValidationResponse(valid=False)

def verify_initial_device_secret(secret: str) -> bool:
    """
    Verify the initial device secret.
    
    Args:
        secret: Secret to verify
        
    Returns:
        True if valid, False otherwise
    """
    return secret == INITIAL_DEVICE_SECRET

def create_device_token(device_id: str, device_name: str) -> Tuple[str, datetime, str]:
    """
    Create a new long-lived device token.
    
    Args:
        device_id: Unique identifier for the device
        device_name: Human-readable name for the device
        
    Returns:
        Tuple of (token, expiration datetime, jwt id)
    """
    expire = datetime.utcnow() + timedelta(days=DEVICE_TOKEN_EXPIRE_DAYS)
    
    # Generate a unique JWT ID
    jti = str(uuid.uuid4())
    
    to_encode = {
        "sub": device_id,
        "device_name": device_name,
        "exp": expire,
        "iat": datetime.utcnow(),
        "jti": jti,
        "token_type": "device",
        "scopes": ["chat:read", "chat:write"]  # Default scopes for devices
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt, expire, jti

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

def revoke_token(token: str) -> bool:
    """
    Revoke a token by adding it to the blacklist.
    
    Args:
        token: Token to revoke
        
    Returns:
        True if token was successfully revoked, False otherwise
    """
    try:
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Get token details
        jti = payload.get("jti")
        exp = payload.get("exp")
        token_type = payload.get("token_type")
        
        if not jti:
            return False
        
        # Calculate TTL (time to live) if token has expiration
        ttl = None
        if exp:
            now = datetime.utcnow()
            exp_datetime = datetime.fromtimestamp(exp)
            if exp_datetime > now:
                ttl = int((exp_datetime - now).total_seconds())
        
        # Add to blacklist
        blacklist_token(jti, ttl)
        
        # If it's a refresh token, also revoke the associated access token
        if token_type == "refresh" and jti in refresh_token_jti_map:
            blacklist_token(refresh_token_jti_map[jti])
            del refresh_token_jti_map[jti]
            
        return True
    except JWTError:
        return False

# Rest of your functions...
# create_tokens, get_client_credentials, validate_token_request, etc.