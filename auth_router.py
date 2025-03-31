from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
import os

from auth import create_device_token, validate_device_token, revoke_device_token, INITIAL_DEVICE_SECRET
from firestore_manager import FirestoreManager

router = APIRouter()
firestore = FirestoreManager()

class DeviceRegistration(BaseModel):
    device_id: str
    device_name: str
    initial_secret: str
    scopes: Optional[List[str]] = None

class TokenData(BaseModel):
    access_token: str
    token_type: str

@router.post("/device", response_model=TokenData)
async def register_device(registration: DeviceRegistration):
    """Register a new device and return a token"""
    if not INITIAL_DEVICE_SECRET or registration.initial_secret != INITIAL_DEVICE_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid initial secret",
        )
    
    # Check if device has already used the initial secret
    device = await firestore.get_device(registration.device_id)
    if device and device.get("has_used_initial_secret", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Initial secret has already been used for this device",
        )
    
    # Register the device
    await firestore.add_or_update_device(
        registration.device_id, 
        {
            "device_name": registration.device_name,
            "is_active": True,
            "has_used_initial_secret": True,
            "created_at": firestore.server_timestamp(),
            "updated_at": firestore.server_timestamp()
        }
    )
    
    # Create a token for the device
    token_data = await create_device_token(
        registration.device_id, 
        registration.device_name, 
        registration.scopes
    )
    
    return token_data

@router.post("/device/revoke")
async def revoke_device(token: str = Depends(validate_device_token)):
    """Revoke a device token"""
    device_id = token.get("sub")
    
    if not device_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token",
        )
    
    # Revoke the token
    success = await revoke_device_token(token)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke token",
        )
    
    # Deactivate the device
    await firestore.add_or_update_device(
        device_id, 
        {
            "is_active": False,
            "updated_at": firestore.server_timestamp()
        }
    )
    
    return {"message": "Token revoked successfully"}

@router.post("/device/verify")
async def verify_device(token: dict = Depends(validate_device_token)):
    """Verify a device token"""
    return {"valid": True, "device_id": token.get("sub"), "scopes": token.get("scopes")}