"""
Database models and Pydantic schemas for Eva LLM Application.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Boolean, BigInteger, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Import Base from database.py - use this as your single source of truth for Base
from database import Base

# SQLAlchemy Models
class ChatMessage(Base):
    """Database model for chat messages."""
    __tablename__ = "chat_messages"
    __table_args__ = {'extend_existing': True}  # Add this to handle duplicate registrations
    
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    is_user = Column(Boolean, default=True, index=True)
    timestamp = Column(BigInteger, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"ChatMessage(id={self.id}, is_user={self.is_user}, timestamp={self.timestamp})"

class ConversationSummary(Base):
    """Database model for conversation summaries."""
    __tablename__ = "conversation_summary"
    __table_args__ = {'extend_existing': True}  # Add this to handle duplicate registrations
    
    id = Column(Integer, primary_key=True, index=True)
    summary = Column(Text, nullable=False)
    timestamp = Column(BigInteger, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"ConversationSummary(id={self.id}, timestamp={self.timestamp})"

class Note(Base):
    """Database model for notes."""
    __tablename__ = "notes"
    __table_args__ = {'extend_existing': True}  # Add this to handle duplicate registrations
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    timestamp = Column(BigInteger, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"Note(id={self.id}, title={self.title[:20]}..., timestamp={self.timestamp})"

class User(Base):
    """Database model for users."""
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}  # Add this to handle duplicate registrations
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"User(id={self.id}, username={self.username}, email={self.email})"

# Pydantic Models (API schemas)
class MessageRequest(BaseModel):
    """Schema for message requests."""
    conversation_id: Optional[str] = None
    content: str
    model: str = "gemini-2.0-flash"  # Updated to latest model version
    
    class Config:
        json_schema_extra = {"example": {"content": "Hello, how can I help you?"}}

class MessageResponse(BaseModel):
    """Schema for message responses."""
    conversation_id: str
    message: str
    created_at: str
    
    class Config:
        json_schema_extra = {"example": {"conversation_id": "123", "message": "Hi there!", "created_at": "2025-03-26T14:47:21Z"}}

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
        json_schema_extra = {"example": {"username": "johndoe", "email": "john@example.com", "password": "securepassword"}}

class UserResponse(UserBase):
    """Schema for user responses."""
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        orm_mode = True  # Keep the original orm_mode for compatibility
        json_schema_extra = {"example": {"id": 1, "username": "johndoe", "email": "john@example.com", "is_active": True, "created_at": "2025-03-26T14:47:21"}}