import logging
import uuid
import json  # Add this import for JSON handling
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from pydantic import BaseModel
import redis.asyncio as redis  # Add this for async Redis support

from config import config
from models import User
from config import SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS, OAUTH_CLIENTS

logger = logging.getLogger(__name__)

# Add Redis client initialization
redis_client = redis.Redis(
    host=config.REDIS_HOST,  # Make sure these are defined in your config
    port=config.REDIS_PORT,
    db=0,
    decode_responses=True
)

# Add this after your imports and before the OAuth2 schemes

async def get_user_by_username(username: str) -> Optional[User]:
    """
    Get user by username (device ID in our case).
    For a personal assistant, this checks if the device is registered.
    """
    try:
        # Assuming you have Redis configured
        device_data_json = await redis_client.get(f"device:{username}")
        
        if not device_data_json:
            return None
            
        device_data = json.loads(device_data_json)
        
        # Create a User instance with device data
        return User(
            username=username,
            device_id=username,
            device_name=device_data.get("device_name", "Unknown Device"),
            is_verified=device_data.get("verified", False)
        )
    except Exception as e:
        logger.error(f"Error fetching user data: {str(e)}")
        return None

# OAuth2 schemes
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
oauth2_code_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="auth/authorize",
    tokenUrl="auth/token"
)

# Token models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int
    scope: str

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []
    jti: str = None

# For Cloud Run, we need a distributed token storage solution
# This in-memory implementation is only for development
# In production, use Firestore, Redis, or another distributed storage
revoked_tokens = set()
refresh_token_jti_map = {}  # Maps refresh tokens to their JTIs for rotation

def create_tokens(data: dict, scopes: List[str], expires_delta: Optional[timedelta] = None):
    """Create access and refresh tokens with proper expiration and JTIs"""
    access_jti = str(uuid.uuid4())
    refresh_jti = str(uuid.uuid4())
    
    # Store mapping for refresh token rotation
    refresh_token_jti_map[refresh_jti] = access_jti
    
    # Access token payload
    access_expires = expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_expire_time = datetime.utcnow() + access_expires
    access_payload = data.copy()
    access_payload.update({
        "exp": access_expire_time,
        "scopes": scopes,
        "jti": access_jti,
        "token_type": "access"
    })
    
    # Refresh token payload
    refresh_expires = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    refresh_expire_time = datetime.utcnow() + refresh_expires
    refresh_payload = {
        "sub": data.get("sub"),
        "exp": refresh_expire_time,
        "scopes": scopes,
        "jti": refresh_jti,
        "token_type": "refresh"
    }
    
    # Create tokens
    access_token = jwt.encode(access_payload, SECRET_KEY, algorithm="HS256")
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm="HS256")
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": access_expires.total_seconds(),
        "scope": " ".join(scopes)
    }

def verify_token_not_revoked(token_data: TokenData):
    """Verify the token hasn't been revoked"""
    if token_data.jti in revoked_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_authorized_scopes(requested_scopes: List[str], user_scopes: List[str]) -> List[str]:
    """Get intersection of requested scopes and user's authorized scopes"""
    return [scope for scope in requested_scopes if scope in user_scopes]

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
        
        # Verify token type
        if payload.get("token_type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        token_data = TokenData(
            username=username, 
            scopes=payload.get("scopes", []),
            jti=payload.get("jti")
        )
        
        # Verify token hasn't been revoked
        verify_token_not_revoked(token_data)
    except JWTError:
        raise credentials_exception
        
    user = get_user_by_username(token_data.username)
    if user is None:
        raise credentials_exception
        
    return user

def validate_client(client_id: str, client_secret: str, redirect_uri: Optional[str] = None):
    """Validate OAuth client credentials and redirect URI"""
    if client_id not in OAUTH_CLIENTS:
        return False
        
    client = OAUTH_CLIENTS[client_id]
    if client["client_secret"] != client_secret:
        return False
        
    if redirect_uri and redirect_uri not in client["redirect_uris"]:
        return False
        
    return True
