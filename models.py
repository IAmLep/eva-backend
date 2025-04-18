"""
Core Pydantic models representing data structures used throughout the application,
including database representations.
"""

import uuid # Import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, EmailStr, HttpUrl, field_validator

# --- Enums ---

class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    AGENT = "agent"

class MemorySource(str, Enum):
    CORE = "core"
    CONVERSATIONAL = "conversational"
    EVENT = "event"
    SYSTEM = "system"
    EXTERNAL = "external"

class MemoryCategory(str, Enum):
    PERSONAL_INFO = "personal_info"
    PREFERENCE = "preference"
    FACT = "fact"
    GOAL = "goal"
    RELATIONSHIP = "relationship"
    SKILL = "skill"
    PROJECT = "project"
    OTHER = "other"

class ApiKeyScope(str, Enum):
    READ_MEMORY = "memory:read"
    WRITE_MEMORY = "memory:write"
    READ_CONVERSATION = "conversation:read"
    WRITE_CONVERSATION = "conversation:write"
    READ_USER = "user:read"
    ADMIN = "admin"

# --- Core Models ---

class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique user identifier") # Use default factory
    username: str = Field(..., description="Unique username")
    email: EmailStr = Field(..., description="User's email address")
    full_name: Optional[str] = Field(None, description="User's full name")
    disabled: bool = Field(default=False, description="Whether the user account is disabled")
    role: UserRole = Field(default=UserRole.USER, description="User role")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User-specific preferences")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        from_attributes = True

class UserInDB(User):
    hashed_password: str = Field(..., description="Hashed password for the user")
    # Salt for deriving encryption keys from password (if that method is chosen later)
    encryption_salt: Optional[str] = Field(None, description="Salt for user-specific encryption key derivation")


class Memory(BaseModel):
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the memory") # Use default factory
    user_id: str = Field(..., description="ID of the user this memory belongs to")
    content: str = Field(..., description="The textual content of the memory")
    source: MemorySource = Field(..., description="Origin of the memory (core, event, etc.)")
    importance: int = Field(default=5, ge=1, le=10, description="Importance score (1-10)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., category, entity, event_time)")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering and retrieval")
    embedding: Optional[List[float]] = Field(None, description="Optional vector embedding for semantic search")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expiration: Optional[datetime] = Field(None, description="Optional expiration time (especially for events)")
    is_synced: bool = Field(default=True, description="Whether this memory state is synced with clients")

    @field_validator('metadata')
    def check_metadata_fields(cls, v, info): # Use 'info' for Pydantic v2
        data = info.data # Get the underlying data being validated
        source = data.get('source')
        if source == MemorySource.CORE:
            if 'category' not in v:
                v['category'] = MemoryCategory.OTHER.value
            if 'category' in v and v['category'] not in [cat.value for cat in MemoryCategory]:
                 v['category'] = MemoryCategory.OTHER.value
            if 'importance' not in v and 'importance' in data:
                 v['importance'] = data.get('importance')

        elif source == MemorySource.EVENT:
            # Ensure event_time is present and valid ISO format string
            event_time_str = v.get("event_time")
            if not event_time_str:
                 raise ValueError("Event memory metadata must include 'event_time'")
            try:
                 datetime.fromisoformat(str(event_time_str).replace('Z', '+00:00'))
            except (TypeError, ValueError):
                 raise ValueError("Event memory metadata 'event_time' must be a valid ISO 8601 string")

            # Ensure expiration is valid ISO format string if present
            exp_time_str = v.get("expiration")
            if exp_time_str:
                 try:
                      datetime.fromisoformat(str(exp_time_str).replace('Z', '+00:00'))
                 except (TypeError, ValueError):
                      raise ValueError("Event memory metadata 'expiration' must be a valid ISO 8601 string if provided")

            # Copy top-level expiration to metadata if not present
            if 'expiration' not in v and data.get('expiration'):
                 v['expiration'] = data['expiration'].isoformat()

            if 'completed' not in v:
                 v['completed'] = False

        return v

    class Config:
        from_attributes = True


class Conversation(BaseModel):
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique conversation identifier")
    user_id: str = Field(..., description="ID of the user involved")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = Field(None, description="Optional summary of the conversation")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncState(BaseModel):
    user_id: str
    device_id: str
    last_sync_time: datetime


class ApiKey(BaseModel):
    key_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the API key") # Use default factory
    user_id: str = Field(..., description="ID of the user this key belongs to")
    prefix: str = Field(..., description="Non-secret prefix of the key (e.g., for identification)")
    hashed_key: str = Field(..., description="Secure hash of the full API key")
    scopes: List[ApiKeyScope] = Field(default_factory=list, description="Permissions granted to this key")
    description: Optional[str] = Field(None, description="User-provided description for the key")
    is_active: bool = Field(default=True, description="Whether the key is currently active")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = Field(None, description="Timestamp of the last time the key was used")
    expires_at: Optional[datetime] = Field(None, description="Optional expiration time for the key")

    class Config:
        from_attributes = True


class UserRateLimit(BaseModel):
    user_id: str
    last_request_time: datetime
    minute_count: int
    day_count: int