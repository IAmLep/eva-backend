import os
import jwt
import uuid
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import Optional, Dict, Any
from firestore_manager import FirestoreManager

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Use environment variable instead of hardcoded secret
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30  # 30 days

# Get from environment instead of hardcoding
INITIAL_DEVICE_SECRET = os.getenv("INITIAL_DEVICE_SECRET")

async def create_device_token(device_id: str, device_name: str, scopes: list = None) -> Dict[str, str]:
    """Create a JWT token for device authentication"""
    if scopes is None:
        scopes = ["chat:read", "chat:write"]
    
    expires = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    jti = str(uuid.uuid4())
    
    to_encode = {
        "sub": device_id,
        "device_name": device_name,
        "scopes": scopes,
        "exp": expires,
        "jti": jti  # Add JWT ID for revocation tracking
    }
    
    token = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    # Store the token JTI in Firestore for potential revocation
    firestore = FirestoreManager()
    await firestore.add_active_token(device_id, jti, expires)
    
    return {"access_token": token, "token_type": "bearer"}

async def validate_device_token(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Validate a JWT token from a device"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode the token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        device_id = payload.get("sub")
        jti = payload.get("jti")
        
        if device_id is None or jti is None:
            raise credentials_exception
        
        # Check if token is revoked
        firestore = FirestoreManager()
        is_revoked = await firestore.is_token_revoked(jti)
        if is_revoked:
            raise credentials_exception
        
        # Check if device is active
        device = await firestore.get_device(device_id)
        if not device or not device.get("is_active", False):
            raise credentials_exception
            
        return payload
        
    except jwt.PyJWTError:
        raise credentials_exception

async def revoke_device_token(token: str) -> bool:
    """Revoke a JWT token"""
    try:
        # Decode the token
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")
        device_id = payload.get("sub")
        
        if jti is None or device_id is None:
            return False
        
        # Add the token to the revoked tokens list
        firestore = FirestoreManager()
        await firestore.revoke_token(jti, device_id)
        
        return True
        
    except jwt.PyJWTError:
        return False

# Remove unused function as indicated by Gemini
# async def get_current_user function removed