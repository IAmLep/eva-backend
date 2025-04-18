"""
Core Pydantic models representing data structures used throughout the application,
including database representations.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, EmailStr, HttpUrl, field_validator

# --- Enums ---

class UserRole(str, Enum):
    """Enumeration for user roles."""
    USER = "user"
    ADMIN = "admin"
    AGENT = "agent" # If EVA acts as an agent

class MemorySource(str, Enum):
    """Where the memory originated from."""
    CORE = "core" # Long-term, factual memory
    CONVERSATIONAL = "conversational" # Extracted from specific conversation turns
    EVENT = "event" # Time-based, like reminders or appointments
    SYSTEM = "system" # Internal system-generated memories (e.g., user onboarding)
    EXTERNAL = "external" # Imported from external sources

class MemoryCategory(str, Enum):
    """Categories primarily for Core memories."""
    PERSONAL_INFO = "personal_info" # Name, address, DOB, etc.
    PREFERENCE = "preference" # Likes, dislikes, settings
    FACT = "fact" # General knowledge about the user or world relevant to them
    GOAL = "goal" # User's stated goals or objectives
    RELATIONSHIP = "relationship" # Information about user's connections
    SKILL = "skill" # User's skills or abilities
    PROJECT = "project" # Details about projects user is working on
    OTHER = "other" # Catch-all category

class ApiKeyScope(str, Enum):
    """Defines permission scopes for API keys."""
    READ_MEMORY = "memory:read"
    WRITE_MEMORY = "memory:write"
    READ_CONVERSATION = "conversation:read"
    WRITE_CONVERSATION = "conversation:write"
    READ_USER = "user:read"
    ADMIN = "admin" # Full access

# --- Core Models ---

class User(BaseModel):
    """Base User model for general use and API responses (excluding sensitive data)."""
    id: str = Field(..., description="Unique user identifier")
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
        from_attributes = True # Allows creating from ORM objects or dicts with extra fields

class UserInDB(User):
    """User model including sensitive data stored in the database."""
    hashed_password: str = Field(..., description="Hashed password for the user")
    # Add other DB-specific fields if needed, like email_verified, last_login etc.
    # email_verified: bool = False
    # last_login: Optional[datetime] = None


class Memory(BaseModel):
    """Represents a single memory unit stored in the database."""
    memory_id: str = Field(..., description="Unique identifier for the memory")
    user_id: str = Field(..., description="ID of the user this memory belongs to")
    content: str = Field(..., description="The textual content of the memory")
    source: MemorySource = Field(..., description="Origin of the memory (core, event, etc.)")
    importance: int = Field(default=5, ge=1, le=10, description="Importance score (1-10)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., category, entity, event_time)")
    tags: List[str] = Field(default_factory=list, description="Tags for filtering and retrieval")
    embedding: Optional[List[float]] = Field(None, description="Optional vector embedding for semantic search") # Placeholder
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Fields relevant for sync and event handling
    expiration: Optional[datetime] = Field(None, description="Optional expiration time (especially for events)")
    is_synced: bool = Field(default=True, description="Whether this memory state is synced with clients")

    # Ensure metadata has expected fields based on source at validation time?
    @field_validator('metadata')
    def check_metadata_fields(cls, v, values):
        data = values.data # Get the underlying data being validated
        source = data.get('source')
        if source == MemorySource.CORE:
            if 'category' not in v:
                v['category'] = MemoryCategory.OTHER.value # Default category if missing
            # Ensure category is valid enum value if present
            if 'category' in v and v['category'] not in [cat.value for cat in MemoryCategory]:
                 v['category'] = MemoryCategory.OTHER.value
            if 'importance' not in v and 'importance' in data: # Copy top-level importance if not in meta
                 v['importance'] = data.get('importance')

        elif source == MemorySource.EVENT:
            if 'event_time' not in v:
                raise ValueError("Event memory metadata must include 'event_time'")
            if 'expiration' not in v and 'expiration' in data: # Copy top-level expiration
                 v['expiration'] = data.get('expiration').isoformat() if data.get('expiration') else None
            if 'completed' not in v:
                 v['completed'] = False # Default to not completed

        return v

    class Config:
        from_attributes = True


class Conversation(BaseModel):
    """Represents a conversation session."""
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique conversation identifier")
    user_id: str = Field(..., description="ID of the user involved")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    summary: Optional[str] = Field(None, description="Optional summary of the conversation")
    # Messages could be stored separately or embedded (embedding can hit Firestore limits)
    # messages: List[Dict[str, Any]] = Field(default_factory=list) # Example if embedding messages
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SyncState(BaseModel):
    """Represents the synchronization state for a user/device."""
    user_id: str
    device_id: str
    last_sync_time: datetime
    # Add other sync-related info if needed


class ApiKey(BaseModel):
    """Represents an API key stored in the database."""
    key_id: str = Field(..., description="Unique identifier for the API key")
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
    """Tracks API usage rate limiting for a user."""
    user_id: str
    last_request_time: datetime
    minute_count: int
    day_count: int
    # Add other tracking periods if needed