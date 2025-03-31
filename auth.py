from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Dict, Any, Optional
import jwt
from datetime import datetime, timedelta
import os
import uuid
import logging
from logging_config import setup_logging
from firestore_manager import firestore_manager

# Setup logging
logger = logging.getLogger(__name__)

# Token-related settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not SECRET_KEY:
    logger.error("JWT_SECRET_KEY environment variable not set")
    raise ValueError("JWT_SECRET_KEY environment variable not set")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

# Initial device secret for device registration
INITIAL_DEVICE_SECRET = os.getenv("INITIAL_DEVICE_SECRET", "")
if not INITIAL_DEVICE_SECRET:
    logger.warning("INITIAL_DEVICE_SECRET environment variable not set, generating a random one")
    INITIAL_DEVICE_SECRET = str(uuid.uuid4())

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

async def create_device_token(device_id: str, scopes: list = None, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Create a JWT token for a device"""
    try:
        if scopes is None:
            scopes = ["chat:read", "chat:write"]
            
        # Set token expiration time
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta
        
        # Create token payload with unique JTI (JWT ID)
        token_jti = str(uuid.uuid4())
        
        token_data = {
            "sub": device_id,
            "scope": " ".join(scopes),
            "exp": expire,
            "jti": token_jti,
        }
        
        # Add user_id if provided
        if user_id:
            token_data["user_id"] = user_id
            
        # Encode token
        encoded_jwt = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        # Store token in Firestore for revocation checking
        await firestore_manager.add_active_token(token_jti, device_id, expire)
        
        return {
            "access_token": encoded_jwt,
            "token_type": "bearer",
            "expires_at": expire.isoformat(),
            "device_id": device_id,
            "scopes": scopes
        }
    except Exception as e:
        logger.error(f"Error creating device token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create device token: {str(e)}"
        )

async def validate_device_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Validate a JWT token and return the device information"""
    if not token:
        logger.warning("No token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Extract token data
        device_id = payload.get("sub")
        scopes = payload.get("scope", "").split()
        token_jti = payload.get("jti")
        user_id = payload.get("user_id")
        
        if not device_id or not token_jti:
            logger.warning("Invalid token payload: missing device_id or token_jti")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if token has been revoked
        is_revoked = await firestore_manager.is_token_revoked(token_jti)
        if is_revoked:
            logger.warning(f"Token {token_jti[:8]}... has been revoked")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Check if device exists and is active
        device = await firestore_manager.get_device(device_id)
        if not device:
            logger.warning(f"Device {device_id} not found")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not device.get("is_active", True):
            logger.warning(f"Device {device_id} is not active")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Device is not active",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Return device information
        return {
            "device_id": device_id,
            "scopes": scopes,
            "user_id": user_id,
            "token_jti": token_jti
        }
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except jwt.DecodeError:
        logger.warning("Token decode error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token decode error",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    except Exception as e:
        logger.error(f"Unexpected error validating token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token validation error: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def revoke_device_token(token: str) -> Dict[str, Any]:
    """Revoke a JWT token"""
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Extract token JTI and device ID
        token_jti = payload.get("jti")
        device_id = payload.get("sub")
        
        if not token_jti or not device_id:
            logger.warning("Invalid token payload during revocation: missing token_jti or device_id")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token payload",
            )
        
        # Add token to revoked list
        await firestore_manager.revoke_token(token_jti)
        
        logger.info(f"Token {token_jti[:8]}... for device {device_id} revoked")
        return {"message": "Token revoked successfully"}
        
    except jwt.ExpiredSignatureError:
        # If token is already expired, just log it
        logger.info("Attempted to revoke already expired token")
        return {"message": "Token already expired"}
        
    except jwt.InvalidTokenError:
        logger.warning("Invalid token during revocation")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token format",
        )
        
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {str(e)}",
        )

async def get_device_from_token(token_data: Dict[str, Any]) -> Dict[str, Any]:
    """Get device details from token data"""
    device_id = token_data.get("device_id")
    if not device_id:
        logger.warning("No device_id in token data")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token data",
        )
    
    device = await firestore_manager.get_device(device_id)
    if not device:
        logger.warning(f"Device {device_id} not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Device not found",
        )
    
    return device

def verify_scope(required_scope: str, token_data: Dict[str, Any]) -> bool:
    """Verify if token has the required scope"""
    scopes = token_data.get("scopes", [])
    return required_scope in scopes