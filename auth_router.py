from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import uuid

from auth import (
    validate_device_token, 
    create_device_token, 
    revoke_device_token, 
    INITIAL_DEVICE_SECRET
)
from firestore_manager import firestore_manager
from api_tools import verify_device

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

class DeviceRegistration(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    device_type: Optional[str] = None
    initial_secret: str

class DeviceVerification(BaseModel):
    device_id: str
    user_id: str

class TokenData(BaseModel):
    access_token: str
    token_type: str
    expires_at: str
    device_id: str
    scopes: List[str]

class TokenRevokeRequest(BaseModel):
    token: str

@router.post("/device", response_model=TokenData)
async def register_device(registration: DeviceRegistration):
    """Register a new device and return a token"""
    try:
        # First check if device already exists
        device = await firestore_manager.get_device(registration.device_id)
        
        # If device exists, check if it has already used the initial secret
        if device and device.get("has_used_initial_secret", False):
            logger.warning(f"Device {registration.device_id} attempting to reuse initial secret")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Initial secret has already been used for this device",
            )
        
        # Validate the initial secret
        if not INITIAL_DEVICE_SECRET or registration.initial_secret != INITIAL_DEVICE_SECRET:
            logger.warning(f"Invalid initial secret provided for device {registration.device_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid initial secret provided",
            )
        
        # Create or update device in Firestore
        device_data = {
            "id": registration.device_id,
            "name": registration.device_name or f"Device {registration.device_id[:8]}",
            "type": registration.device_type or "unknown",
            "is_active": True,
            "has_used_initial_secret": True,
            "created_at": device.get("created_at") if device else None,
            "last_seen_at": None
        }
        
        # Save device to Firestore
        await firestore_manager.add_or_update_device(
            registration.device_id, 
            device_data
        )
        
        # Create a token for the device
        token_data = await create_device_token(registration.device_id)
        
        logger.info(f"Device {registration.device_id} registered successfully")
        return token_data
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error registering device: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register device: {str(e)}",
        )

@router.post("/device/revoke")
async def revoke_device(request: TokenRevokeRequest):
    """Revoke a device token"""
    try:
        result = await revoke_device_token(request.token)
        
        # Extract device_id from token for device deactivation
        # This is a simplified approach; in a real system you might want to 
        # decode the token first to get the device_id
        try:
            from jwt import decode, InvalidTokenError
            from auth import SECRET_KEY, ALGORITHM
            
            payload = decode(request.token, SECRET_KEY, algorithms=[ALGORITHM])
            device_id = payload.get("sub")
            
            if device_id:
                # Deactivate the device
                device = await firestore_manager.get_device(device_id)
                if device:
                    device["is_active"] = False
                    await firestore_manager.add_or_update_device(device_id, device)
                    logger.info(f"Device {device_id} deactivated")
        except InvalidTokenError:
            logger.warning("Could not decode token for device deactivation")
        except Exception as e:
            logger.error(f"Error deactivating device: {str(e)}")
        
        logger.info("Token revoked successfully")
        return {"message": "Token revoked successfully"}
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error revoking token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke token: {str(e)}",
        )

@router.post("/verify")
async def verify_association(
    verification: DeviceVerification,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Associate a device with a user"""
    try:
        # Verify that the token is for the device being verified
        if token_data["device_id"] != verification.device_id:
            logger.warning(f"Token device_id {token_data['device_id']} doesn't match request device_id {verification.device_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token does not match the device being verified",
            )
        
        # Get device
        device = await firestore_manager.get_device(verification.device_id)
        if not device:
            logger.warning(f"Device {verification.device_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found",
            )
        
        # Update device with user_id
        device["user_id"] = verification.user_id
        device["verified_at"] = uuid.uuid4().hex  # Generate a unique timestamp
        
        # Save updated device
        await firestore_manager.add_or_update_device(verification.device_id, device)
        
        # Create a new token with user_id included
        new_token = await create_device_token(
            verification.device_id, 
            scopes=token_data.get("scopes", ["chat:read", "chat:write"]),
            user_id=verification.user_id
        )
        
        logger.info(f"Device {verification.device_id} associated with user {verification.user_id}")
        return {
            "message": "Device verified and associated with user",
            "token": new_token
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error verifying device: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify device: {str(e)}",
        )

@router.get("/device/{device_id}")
async def get_device_info(
    device_id: str,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Get information about a device"""
    try:
        # Check if the token is for the requested device or admin access
        if token_data["device_id"] != device_id and "admin" not in token_data.get("scopes", []):
            logger.warning(f"Token device_id {token_data['device_id']} doesn't match request device_id {device_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this device information",
            )
        
        # Get device
        device = await firestore_manager.get_device(device_id)
        if not device:
            logger.warning(f"Device {device_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found",
            )
        
        # Remove sensitive information
        if "has_used_initial_secret" in device:
            del device["has_used_initial_secret"]
        
        logger.info(f"Retrieved info for device {device_id}")
        return device
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error getting device info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get device info: {str(e)}",
        )