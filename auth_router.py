import logging
import secrets
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from jose import JWTError
import jwt
import redis.asyncio as redis
from fastapi import APIRouter, HTTPException, Depends, Form, Request, Response, status, Header
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import config
from schemas import AuthRequest
from auth import (
    verify_token, create_access_token, create_refresh_token, Token,
    create_device_token, verify_initial_device_secret, validate_device_token,
    DeviceAuthRequest, DeviceToken, DeviceValidationResponse, revoke_token
)
from rate_limiter import limiter, rate_limit
from database import get_db, Device
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter()

# Create a Redis client instance.
redis_client = redis.Redis(
    host=config.REDIS_HOST, port=config.REDIS_PORT, db=0, decode_responses=True
)

# Apply rate limiting to auth endpoints
auth_limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

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

@router.post("/device/auth", response_model=DeviceToken)
@rate_limit(limit=3, period=60)  # Strict rate limiting for authentication
async def authenticate_device(request: DeviceAuthRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticate a device using the initial secret and issue a long-lived device token.
    """
    logger.info(f"Device authentication request received for: {request.device_id}")
    
    # Validate initial secret
    if not verify_initial_device_secret(request.initial_secret):
        logger.warning(f"Invalid initial secret used for device: {request.device_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid initial device secret"
        )
    
    # Generate device token
    device_token, expire, jti = create_device_token(
        device_id=request.device_id,
        device_name=request.device_name
    )
    
    # Store device info in database
    device = await db.query(Device).filter(Device.device_id == request.device_id).first()
    
    if not device:
        device = Device(
            device_id=request.device_id,
            device_name=request.device_name,
            device_model=request.device_model,
            token_jti=jti
        )
        db.add(device)
    else:
        device.device_name = request.device_name
        device.last_sync = datetime.utcnow()
        device.token_jti = jti
        if request.device_model:
            device.device_model = request.device_model
    
    await db.commit()
    
    logger.info(f"Device authenticated successfully: {request.device_id}")
    
    # Calculate expires_in in seconds
    expires_in = int((expire - datetime.utcnow()).total_seconds())
    
    return DeviceToken(
        device_token=device_token,
        token_type="device",
        expires_at=expire,
        expires_in=expires_in,
        device_id=request.device_id
    )

@router.post("/device/validate", response_model=DeviceValidationResponse)
async def validate_device(
    token: str = Header(..., description="Device token to validate")
):
    """
    Validate a device token and return its status.
    """
    # Remove Bearer prefix if present
    if token.startswith("Bearer "):
        token = token[7:]
    
    validation_result = validate_device_token(token)
    return validation_result

@router.post("/device/revoke", status_code=status.HTTP_204_NO_CONTENT)
@rate_limit(limit=5, period=60)
async def revoke_device_token(
    token: str = Header(..., description="Device token to revoke"),
    db: AsyncSession = Depends(get_db)
):
    """
    Revoke a device token.
    """
    # First validate the token
    validation = validate_device_token(token)
    
    if not validation.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token"
        )
    
    # Revoke the token
    success = revoke_token(token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke token"
        )
    
    # Update device record if found
    if validation.device_id:
        device = await db.query(Device).filter(Device.device_id == validation.device_id).first()
        if device:
            device.is_active = False
            await db.commit()
    
    return Response(status_code=status.HTTP_204_NO_CONTENT)

@router.post("/register")
@rate_limit(limit=3, period=60)
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
