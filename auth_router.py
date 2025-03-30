import logging
import secrets
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from jose import JWTError
import jwt
from fastapi import APIRouter, HTTPException, Depends, Form, Request, Response, status, Header
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import SECRET_KEY, JWT_ALGORITHM, VERIFICATION_TIMEOUT
from config import ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
from schemas import AuthRequest
from auth import (
    verify_token, create_access_token, create_refresh_token, Token,
    create_device_token, verify_initial_device_secret, validate_device_token,
    DeviceAuthRequest, DeviceToken, DeviceValidationResponse, revoke_token
)
from rate_limiter import limiter, rate_limit
from database import get_db, Device
from sqlalchemy.orm import Session
from firestore_manager import store_document, get_document, delete_document

logger = logging.getLogger(__name__)

router = APIRouter()

# Apply rate limiting to auth endpoints
auth_limiter = Limiter(key_func=get_remote_address, default_limits=["5/minute"])

# Collection names for Firestore
DEVICES_COLLECTION = "devices"
VERIFICATIONS_COLLECTION = "verifications"
TOKENS_COLLECTION = "tokens"

# Helper function to store device data in Firestore
async def store_device_data(device_id: str, data: Dict[str, Any]) -> None:
    try:
        # Include timestamp for TTL if needed
        data["updated_at"] = datetime.utcnow().isoformat()
        await store_document(DEVICES_COLLECTION, device_id, data)
        logger.info(f"Stored device data for device: {device_id}")
    except Exception as e:
        logger.error(f"Error storing device data: {e}")

# Helper function to get device data from Firestore
async def get_device_data(device_id: str) -> Optional[Dict[str, Any]]:
    try:
        data = await get_document(DEVICES_COLLECTION, device_id)
        return data
    except Exception as e:
        logger.error(f"Error retrieving device data: {e}")
        return None

# Helper function to store verification code with expiry
async def store_verification_code(device_id: str, code: str, expires_in_seconds: int) -> None:
    try:
        expiry = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
        data = {
            "code": code,
            "expires_at": expiry.isoformat(),
            "created_at": datetime.utcnow().isoformat()
        }
        await store_document(VERIFICATIONS_COLLECTION, device_id, data)
        logger.info(f"Stored verification code for device: {device_id}")
    except Exception as e:
        logger.error(f"Error storing verification code: {e}")

# Helper function to get and validate verification code
async def get_verification_code(device_id: str) -> Optional[str]:
    try:
        data = await get_document(VERIFICATIONS_COLLECTION, device_id)
        if not data:
            return None
            
        # Check if expired
        expires_at = datetime.fromisoformat(data["expires_at"])
        if datetime.utcnow() > expires_at:
            # Clean up expired code
            await delete_document(VERIFICATIONS_COLLECTION, device_id)
            return None
            
        return data["code"]
    except Exception as e:
        logger.error(f"Error retrieving verification code: {e}")
        return None

# Helper function to store token data
async def store_token_data(token_type: str, device_id: str, token: str, expires_in_seconds: int) -> None:
    try:
        expiry = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
        data = {
            "token": token,
            "expires_at": expiry.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "device_id": device_id,
            "type": token_type
        }
        doc_id = f"{token_type}:{device_id}"
        await store_document(TOKENS_COLLECTION, doc_id, data)
        logger.info(f"Stored {token_type} token for device: {device_id}")
    except Exception as e:
        logger.error(f"Error storing token data: {e}")

# Helper function to get token data
async def get_token_data(token_type: str, device_id: str) -> Optional[Dict[str, Any]]:
    try:
        doc_id = f"{token_type}:{device_id}"
        data = await get_document(TOKENS_COLLECTION, doc_id)
        if not data:
            return None
            
        # Check if expired
        expires_at = datetime.fromisoformat(data["expires_at"])
        if datetime.utcnow() > expires_at:
            # Clean up expired token
            await delete_document(TOKENS_COLLECTION, doc_id)
            return None
            
        return data
    except Exception as e:
        logger.error(f"Error retrieving token data: {e}")
        return None

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

# UPDATED PATH: Changed from /device/auth to /device to match versioned path
@router.post("/device", response_model=DeviceToken)
@rate_limit(limit=3, period=60)  # Strict rate limiting for authentication
async def authenticate_device(request: DeviceAuthRequest, db: Session = Depends(get_db)):
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
    device = db.query(Device).filter(Device.device_id == request.device_id).first()
    
    if not device:
        # Only use fields that exist in the Device model
        device = Device(
            device_id=request.device_id,
            device_name=request.device_name
        )
        db.add(device)
    else:
        device.device_name = request.device_name
        device.last_sync = datetime.utcnow()
    
    db.commit()
    
    # Store token JTI in Firestore
    await store_document(TOKENS_COLLECTION, f"jti:{request.device_id}", {
        "jti": jti,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": expire.isoformat()
    })
    
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

# Paths for these remaining endpoints are fine as they already align with what we expect
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
    db: Session = Depends(get_db)
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
        device = db.query(Device).filter(Device.device_id == validation.device_id).first()
        if device:
            device.is_active = False
            db.commit()
    
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
    
    # Store the code with expiry
    await store_verification_code(device.device_id, verification_code, VERIFICATION_TIMEOUT)
    
    # Store device info
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
        "expires_in": VERIFICATION_TIMEOUT
    }

@router.post("/verify", response_model=Token)
async def verify_device(verification: VerificationRequest, db: Session = Depends(get_db)):
    """
    Verify the provided code and issue OAuth2 tokens
    """
    stored_code = await get_verification_code(verification.device_id)
    
    if not stored_code or stored_code != verification.code:
        raise HTTPException(status_code=401, detail="Invalid verification code")
    
    # Mark the device as verified
    device_data = await get_device_data(verification.device_id) or {}
    device_data["verified"] = True
    await store_device_data(verification.device_id, device_data)
    
    # Clean up the verification code
    await delete_document(VERIFICATIONS_COLLECTION, verification.device_id)
    
    # Register device in database if not exists
    device = db.query(Device).filter(Device.device_id == verification.device_id).first()
    if not device:
        device = Device(
            device_id=verification.device_id, 
            device_name=f"Device-{verification.device_id[:8]}"
        )
        db.add(device)
        db.commit()
    
    # Generate tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, _, _ = create_access_token(
        data={"sub": verification.device_id, "scopes": ["chat:read", "chat:write"]},
        expires_delta=access_token_expires
    )
    
    refresh_token, _ = create_refresh_token(
        data={"sub": verification.device_id}
    )
    
    # Store refresh token with expiry
    await store_token_data(
        "refresh", 
        verification.device_id, 
        refresh_token, 
        REFRESH_TOKEN_EXPIRE_DAYS * 86400  # Convert days to seconds
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
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
    device_data = await get_device_data(device_id)
    
    if not device_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Device not registered",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not device_data.get("verified", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Device not verified",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token, _, _ = create_access_token(
        data={"sub": device_id, "scopes": form_data.scopes},
        expires_delta=access_token_expires
    )
    
    refresh_token, _ = create_refresh_token(
        data={"sub": device_id}
    )
    
    # Store refresh token with expiry
    await store_token_data(
        "refresh", 
        device_id, 
        refresh_token, 
        REFRESH_TOKEN_EXPIRE_DAYS * 86400  # Convert days to seconds
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
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
            SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        
        device_id = payload.get("sub")
        token_type = payload.get("type")
        
        if not device_id or token_type != "refresh":
            raise credentials_exception
        
        # Verify refresh token against stored token
        token_data = await get_token_data("refresh", device_id)
        
        if not token_data or token_data.get("token") != refresh_request.refresh_token:
            raise credentials_exception
        
        # Generate new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token, _, _ = create_access_token(
            data={"sub": device_id, "scopes": payload.get("scopes", ["chat:read", "chat:write"])},
            expires_delta=access_token_expires
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_request.refresh_token  # Return the same refresh token
        )
        
    except JWTError:
        raise credentials_exception