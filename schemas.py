"""
Pydantic schemas for API request/response validation and serialization.

These models define the expected structure of data sent to and received from API endpoints.
They often subset or transform the core data models defined in models.py.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, EmailStr, field_validator

# Import enums from models if needed
from models import UserRole, MemorySource, MemoryCategory, ApiKeyScope

# --- User Schemas ---

class UserBase(BaseModel):
    """Base schema for user data, shared properties."""
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="User's email address")
    full_name: Optional[str] = Field(None, max_length=100, description="User's full name")

class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str = Field(..., min_length=8, description="User's password")

    @field_validator('password')
    def password_complexity(cls, v):
        """Example password complexity validation."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        # Add more rules (uppercase, digit, symbol) if desired
        # if not re.search(r"[A-Z]", v): raise ValueError("Password must contain an uppercase letter")
        # if not re.search(r"[0-9]", v): raise ValueError("Password must contain a digit")
        return v

class UserUpdate(BaseModel):
    """Schema for updating user details (optional fields)."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    preferences: Optional[Dict[str, Any]] = None
    # Add other updatable fields as needed

class UserResponse(UserBase):
    """Schema for returning user data in API responses (excluding password)."""
    id: str = Field(..., description="Unique user identifier")
    is_active: bool = Field(..., description="Whether the user account is active") # Renamed from 'disabled'
    role: UserRole = Field(..., description="User role")
    preferences: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True # Allow creating from User model instance

# --- Token Schemas ---

class Token(BaseModel):
    """Schema for representing JWT access token response."""
    access_token: str
    token_type: str = "bearer"
    expires_at: Optional[datetime] = None # Include expiration time

# Inherit from Token for LoginResponse if preferred, or keep separate as in auth_router.py
# class LoginResponse(Token):
#     user: UserResponse

# --- Memory Schemas ---
# Schemas for API interaction with memories might be needed if exposing memory CRUD via API
# Example:
class MemoryCreateAPI(BaseModel):
    """Schema for creating a memory via API (if exposed)."""
    content: str
    source: MemorySource
    importance: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    expiration: Optional[datetime] = None

class MemoryResponseAPI(BaseModel):
    """Schema for returning memory data via API."""
    memory_id: str
    user_id: str
    content: str
    source: MemorySource
    importance: int
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    expiration: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- API Key Schemas (if managing keys via API) ---
class ApiKeyCreate(BaseModel):
    """Schema for requesting API key creation."""
    description: Optional[str] = Field(None, max_length=100)
    scopes: List[ApiKeyScope] = Field(default=[ApiKeyScope.READ_MEMORY, ApiKeyScope.WRITE_CONVERSATION]) # Example default
    expires_at: Optional[datetime] = None

class ApiKeyResponse(BaseModel):
    """Schema for returning API key details (EXCLUDING the secret key itself)."""
    key_id: str
    prefix: str # Show only the prefix
    user_id: str
    description: Optional[str] = None
    scopes: List[ApiKeyScope]
    is_active: bool
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ApiKeyCreateResponse(ApiKeyResponse):
    """Schema returned immediately after creating a key (includes the full key ONCE)."""
    full_key: str = Field(..., description="The full API key. Store this securely, it will not be shown again.")

# --- Add other schemas as needed for secrets, sync, etc. ---
# Schemas for secrets_router are currently defined within that file,
# but could be moved here for consistency if desired.