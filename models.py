"""
Pydantic schemas and Pydantic models for API requests/responses.
SQLAlchemy database models are defined in database.py.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field

# Pydantic Models (API schemas)
class MessageRequest(BaseModel):
    """Schema for message requests."""
    conversation_id: Optional[str] = None
    content: str
    model: str = "gemini-2.0-flash"  # Updated to latest model version
    
    class Config:
        schema_extra = {"example": {"content": "Hello, how can I help you?"}}

class MessageResponse(BaseModel):
    """Schema for message responses."""
    conversation_id: str
    message: str
    created_at: str
    
    class Config:
        schema_extra = {"example": {"conversation_id": "123", "message": "Hi there!", "created_at": "2025-03-26T14:47:21Z"}}

class ConversationRequest(BaseModel):
    """Schema for conversation requests."""
    id: str
    title: str
    created_at: str
    updated_at: str

class DeviceInfo(BaseModel):
    """Schema for device information."""
    device_id: str
    device_name: Optional[str] = None
    last_sync: Optional[str] = None

class UserBase(BaseModel):
    """Base schema for user data."""
    username: str
    email: Optional[str] = None

class UserCreate(UserBase):
    """Schema for user creation."""
    password: str
    
    class Config:
        schema_extra = {"example": {"username": "johndoe", "email": "john@example.com", "password": "securepassword"}}

class UserResponse(UserBase):
    """Schema for user responses."""
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True  # Older version of Pydantic
        schema_extra = {"example": {"id": 1, "username": "johndoe", "email": "john@example.com", "is_active": True, "created_at": "2025-03-26T14:47:21"}}