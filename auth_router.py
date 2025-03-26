import logging
import secrets
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from jose import JWTError
import jwt
import redis.asyncio as redis
from fastapi import APIRouter, HTTPException, Depends, Form, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from config import config
from schemas import AuthRequest
from auth import create_access_token, create_refresh_token, Token
from rate_limiter import limiter
from database import get_db, Device
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter()

# Create a Redis client instance.
redis_client = redis.Redis(
    host=config.REDIS_HOST, port=config.REDIS_PORT, db=0, decode_responses=True
)

# Helper function to store device data.
async def store_device_data(device_id: str, data: Dict[str, Any]) -> None:
    try:
        await redis_client.set(f"device:{device_id}", json.dumps(data))
        logger.info(f"Stored device data for device: {device_id}")
    except Exception as e:
        logger.error(f"Error storing device data: {e}")

# Model for device registration
class DeviceRegistration(BaseModel):
    device_id: str
    device_name: Optional[str] = None

# Model for verification request
class VerificationRequest(BaseModel):
    device_id: str
    code: str

# Model for token refresh
class RefreshRequest(BaseModel):
    refresh_token: str

@router.post("/register")
async def register_device(device: DeviceRegistration) -> Dict[str, Any]:
    """
    Register a new device and generate a verification code
    """
    logger.info(f"Registration request received for device: {device.device_id}")
    
    # Generate a verification code
    verification_code = "".join([str(secrets.randbelow(10)) for _ in range(6)])
    
    # Store the code in Redis with expiry
    await redis_client.setex(
        f"verify:{device.device_id}",
        config.VERIFICATION_TIMEOUT,
        verification_code
    )
    
    # Store device info in Redis
    await store_device_data(
        device.device_id, 
        {
            "device_name": device.device_name or f"Device-{device.device_id[:8]}",
            "verified": False,
            "registration_time": datetime.utcnow().isoformat()
        }
    )
    
    logger.info(f"Generated verification code {verification_code} for device {device.device_id}")
    return {
        "requires_verification": True,
        "verification_code": verification_code,  # In production, send via a separate channel like SMS
        "expires_in": config.VERIFICATION_TIMEOUT
    }

@router.post("/verify", response_model=Token)
async def verify_device(verification: VerificationRequest, db: AsyncSession = Depends(get_db)) -> Token:
    """
    Verify the provided code and issue OAuth2 tokens
    """
    stored_code = await redis_client.get(f"verify:{verification.device_id}")
    if not stored_code or stored_code != verification.code:
        raise HTTPException(status_code=401, detail="Invalid verification code")
    
    # Mark the device as verified in Redis
    await store_device_data(verification.device_id, {"verified": True})
    
    # Clean up the verification code
    await redis_client.delete(f"verify:{verification.device_id}")
    
    # Register device in database if not exists
    device = await db.query(Device).filter(Device.device_id == verification.device_id).first()
    if not device:
        device = Device(
            device_id=verification.device_id, 
            device_name=f"Device-{verification.device_id[:8]}"
        )
        db.add(device)
        await db.commit()
    
    # Generate tokens
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": verification.device_id, "scopes": ["chat:read", "chat:write"]},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": verification.device_id}
    )
    
    # Store refresh token in Redis for later validation
    await redis_client.setex(
        f"refresh:{verification.device_id}",
        config.REFRESH_TOKEN_EXPIRE_DAYS * 86400,  # Convert days to seconds
        refresh_token
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        refresh_token=refresh_token
    )

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> Token:
    """
    OAuth2 compatible token endpoint for password flow (development/testing)
    """
    # In a real implementation, you would validate username/password
    # For this simplified version, we'll check if the device is verified
    device_id = form_data.username
    device_data_json = await redis_client.get(f"device:{device_id}")
    
    if not device_data_json:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Device not registered",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    device_data = json.loads(device_data_json)
    if not device_data.get("verified", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Device not verified",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate tokens
    access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": device_id, "scopes": form_data.scopes},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": device_id}
    )
    
    # Store refresh token
    await redis_client.setex(
        f"refresh:{device_id}",
        config.REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        refresh_token
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        refresh_token=refresh_token
    )

@router.post("/refresh", response_model=Token)
async def refresh_access_token(refresh_request: RefreshRequest) -> Token:
    """
    Endpoint to refresh an access token using a refresh token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode token to get device_id
        payload = jwt.decode(
            refresh_request.refresh_token, 
            config.SECRET_KEY, 
            algorithms=[config.JWT_ALGORITHM]
        )
        
        device_id = payload.get("sub")
        token_type = payload.get("type")
        
        if not device_id or token_type != "refresh":
            raise credentials_exception
        
        # Verify refresh token against stored token
        stored_refresh_token = await redis_client.get(f"refresh:{device_id}")
        if not stored_refresh_token or stored_refresh_token != refresh_request.refresh_token:
            raise credentials_exception
        
        # Generate new access token
        access_token_expires = timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": device_id, "scopes": payload.get("scopes", ["chat:read", "chat:write"])},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_request.refresh_token  # Return the same refresh token
        )
        
    except JWTError:
        raise credentials_exception

import uuid
import secrets
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Form, Query, Request
from fastapi.security import OAuth2PasswordRequestForm
from jose import JWTError, jwt
from pydantic import BaseModel

from auth import (
    create_tokens, get_current_user, validate_client, 
    revoked_tokens, refresh_token_jti_map
)
# Option 1 (if User is in database.py):
from database import User
# OR Option 2 (if User is in models.py):
from models import User

# You might need to define or import authenticate_user and get_user_by_username
# from wherever they're actually defined
from config import SECRET_KEY, OAUTH_CLIENTS

router = APIRouter(prefix="/auth", tags=["authentication"])

# New models for OAuth flows
class AuthorizationRequest(BaseModel):
    response_type: str
    client_id: str
    redirect_uri: Optional[str] = None
    scope: Optional[str] = None
    state: Optional[str] = None

class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

# In-memory storage - for production, use Firestore or other GCP storage service
device_codes = {}  # Maps device_code to user_code
auth_codes = {}    # Maps authorization codes to data

# Add this function before your router endpoints

async def authenticate_user(username: str, password: str):
    """
    Authenticates a user by username and password.
    For a personal assistant, this could validate device credentials.
    """
    # For a personal app, you might want to implement a simplified authentication
    # that works with registered devices rather than username/password
    
    # Example implementation for device-based auth:
    device_data_json = await redis_client.get(f"device:{username}")
    
    if not device_data_json:
        return None
    
    device_data = json.loads(device_data_json)
    
    # For a personal assistant, we might just check if the device is verified
    # rather than a traditional password check
    if device_data.get("verified", False):
        # Return a simple user object with the device ID as username
        return type('User', (), {'username': username})
    
    return None

@router.post("/token", response_model=dict)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Password grant endpoint for OAuth 2.0"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get requested scopes and validate against allowed scopes
    requested_scopes = form_data.scopes if form_data.scopes else ["chat:read", "chat:write"]
    allowed_scopes = ["chat:read", "chat:write", "profile:read"]  # This should come from user permissions
    scopes = [scope for scope in requested_scopes if scope in allowed_scopes]
    
    return create_tokens({"sub": user.username}, scopes)

@router.post("/refresh", response_model=dict)
async def refresh_access_token(refresh_token: str = Form(...)):
    """Refresh token endpoint with token rotation"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=["HS256"])
        
        # Verify it's a refresh token
        if payload.get("token_type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Check if token is revoked
        jti = payload.get("jti")
        if jti in revoked_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Revoke the old refresh token
        revoked_tokens.add(jti)
        
        # If there was an associated access token, revoke it too
        if jti in refresh_token_jti_map:
            revoked_tokens.add(refresh_token_jti_map[jti])
            del refresh_token_jti_map[jti]
        
        # Create new tokens
        username = payload.get("sub")
        scopes = payload.get("scopes", [])
        
        return create_tokens({"sub": username}, scopes)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/revoke")
async def revoke_token(
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
    client_id: str = Form(...),
    client_secret: str = Form(...)
):
    """Token revocation endpoint (RFC 7009)"""
    # Validate client
    if not validate_client(client_id, client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials"
        )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        jti = payload.get("jti")
        
        if jti:
            revoked_tokens.add(jti)
            
            # If revoking a refresh token, also revoke its access token
            if payload.get("token_type") == "refresh" and jti in refresh_token_jti_map:
                revoked_tokens.add(refresh_token_jti_map[jti])
                del refresh_token_jti_map[jti]
        
        return {"status": "success"}
    except JWTError:
        # Per RFC 7009, invalid tokens should not cause an error
        return {"status": "success"}

@router.get("/authorize")
async def authorize(
    response_type: str = Query(...),
    client_id: str = Query(...),
    redirect_uri: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Authorization endpoint for OAuth 2.0 authorization code flow"""
    if response_type != "code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported response_type"
        )
    
    # Validate client and redirect URI
    if client_id not in OAUTH_CLIENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client_id"
        )
    
    client = OAUTH_CLIENTS[client_id]
    if redirect_uri and redirect_uri not in client["redirect_uris"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid redirect_uri"
        )
    
    # Generate authorization code
    auth_code = secrets.token_urlsafe(32)
    
    # Store code with user and client info
    requested_scopes = scope.split() if scope else []
    allowed_scopes = ["chat:read", "chat:write", "profile:read"]  # Should come from user permissions
    granted_scopes = [s for s in requested_scopes if s in allowed_scopes]
    
    # Store authorization data with 10-minute expiration
    expires_at = datetime.utcnow() + timedelta(minutes=10)
    auth_codes[auth_code] = {
        "client_id": client_id,
        "user_id": current_user.username,
        "scopes": granted_scopes,
        "redirect_uri": redirect_uri,
        "expires_at": expires_at
    }
    
    # Redirect user back to client
    redirect_url = redirect_uri or client["redirect_uris"][0]
    redirect_url += f"?code={auth_code}"
    if state:
        redirect_url += f"&state={state}"
    
    # In a real implementation, return a RedirectResponse
    # For this example, just return the URL
    return {"redirect_to": redirect_url}

@router.post("/token/code")
async def exchange_code_for_token(
    grant_type: str = Form(...),
    code: str = Form(...),
    redirect_uri: Optional[str] = Form(None),
    client_id: str = Form(...),
    client_secret: str = Form(...)
):
    """Exchange authorization code for tokens"""
    if grant_type != "authorization_code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported grant_type"
        )
    
    # Validate client
    if not validate_client(client_id, client_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials"
        )
    
    # Verify code exists and is valid
    if code not in auth_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid authorization code"
        )
    
    code_data = auth_codes[code]
    
    # Check code expiration
    if datetime.utcnow() > code_data["expires_at"]:
        del auth_codes[code]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authorization code expired"
        )
    
    # Verify client_id matches the one used for the code
    if code_data["client_id"] != client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="client_id mismatch"
        )
    
    # Verify redirect_uri if provided
    if redirect_uri and code_data.get("redirect_uri") and redirect_uri != code_data["redirect_uri"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="redirect_uri mismatch"
        )
    
    # Delete the code as it's single-use
    user_id = code_data["user_id"]
    scopes = code_data["scopes"]
    del auth_codes[code]
    
    # Create tokens
    return create_tokens({"sub": user_id}, scopes)

@router.post("/device/code")
async def device_authorization(
    client_id: str = Form(...),
    scope: Optional[str] = Form(None)
):
    """Device Authorization Request (RFC 8628)"""
    # Validate client exists
    if client_id not in OAUTH_CLIENTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client_id"
        )
    
    # Generate device and user codes
    device_code = secrets.token_urlsafe(32)
    user_code = "-".join([secrets.token_hex(3).upper() for _ in range(2)])
    
    # Store the codes with expiration
    expires_in = 1800  # 30 minutes
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    
    device_codes[device_code] = {
        "user_code": user_code,
        "client_id": client_id,
        "scope": scope,
        "expires_at": expires_at,
        "status": "pending"
    }
    
    # The verification URI should be a frontend page where users can enter their code
    verification_uri = f"https://{Request.headers.get('host', 'example.com')}/device-verify"
    
    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": f"{verification_uri}?user_code={user_code}",
        "expires_in": expires_in,
        "interval": 5
    }

@router.post("/device/token")
async def device_token(
    grant_type: str = Form(...),
    device_code: str = Form(...),
    client_id: str = Form(...)
):
    """Device Access Token Request (RFC 8628)"""
    if grant_type != "urn:ietf:params:oauth:grant-type:device_code":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid grant_type"
        )
    
    # Validate device code
    if device_code not in device_codes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid device_code"
        )
    
    device_info = device_codes[device_code]
    
    # Check client_id
    if device_info["client_id"] != client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid client_id for this device_code"
        )
    
    # Check expiration
    if datetime.utcnow() > device_info["expires_at"]:
        # Clean up expired code
        del device_codes[device_code]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="device_code expired"
        )
    
    # Check authorization status
    if device_info["status"] == "pending":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="authorization_pending"
        )
    
    if device_info["status"] == "denied":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="access_denied"
        )
    
    if device_info["status"] == "approved":
        # Get user and scopes from the authorization
        username = device_info["username"]
        requested_scopes = device_info["scope"].split() if device_info["scope"] else []
        allowed_scopes = ["chat:read", "chat:write", "profile:read"]  # Should come from user permissions
        scopes = [scope for scope in requested_scopes if scope in allowed_scopes]
        
        # Clean up after successful authorization
        del device_codes[device_code]
        
        # Create tokens
        return create_tokens({"sub": username}, scopes)

@router.post("/device/approve")
async def approve_device(
    user_code: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """Approve a device authorization request"""
    # Find the device code that matches this user code
    matching_device_code = None
    matching_code_info = None
    
    for device_code, info in device_codes.items():
        if info["user_code"] == user_code:
            matching_device_code = device_code
            matching_code_info = info
            break
    
    if not matching_device_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired user_code"
        )
    
    # Check expiration
    if datetime.utcnow() > matching_code_info["expires_at"]:
        del device_codes[matching_device_code]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Code expired"
        )
    
    # Approve the authorization
    device_codes[matching_device_code]["status"] = "approved"
    device_codes[matching_device_code]["username"] = current_user.username
    
    return {"status": "approved"}
