"""
Models module for EVA backend.

This module defines all data models used throughout the application,
including enhanced memory models for the multi-tiered memory system.

Update your existing models.py file with this version.

Current Date: 2025-04-13
Current User: IAmLep
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, EmailStr, Field, validator


class UserRole(str, Enum):
    """
    User role enumeration.
    
    Defines possible roles for users in the system.
    
    Attributes:
        ADMIN: Administrator with full access
        USER: Standard user
        GUEST: Guest user with limited access
    """
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"


class User(BaseModel):
    """
    User model.
    
    Represents a user in the system with non-sensitive information.
    
    Attributes:
        id: Unique identifier
        username: Unique username
        email: Email address
        full_name: Optional full name
        created_at: Account creation timestamp
        updated_at: Last update timestamp
        disabled: Whether the user is disabled
        role: User role
        preferences: User preferences
        metadata: Additional metadata
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    disabled: bool = False
    role: UserRole = UserRole.USER
    preferences: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class UserInDB(User):
    """
    User model with sensitive information.
    
    Extends the User model with password hash. Used internally
    and should never be returned to clients.
    
    Attributes:
        hashed_password: Hashed password
    """
    hashed_password: str


class MemorySource(str, Enum):
    """
    Memory source enumeration.
    
    Defines the source/type of a memory.
    
    Attributes:
        CORE: Long-term important memories
        CONVERSATION: Conversational memories
        EVENT: Time-based memories like appointments
        SYSTEM: System-generated memories
    """
    CORE = "core"
    CONVERSATION = "conversational"
    EVENT = "event"
    SYSTEM = "system"


class MemoryCategory(str, Enum):
    """
    Memory category enumeration for core memories.
    
    Attributes:
        PERSON: Information about people
        PLACE: Information about places
        PREFERENCE: User preferences and likes/dislikes
        FACT: General factual information
        OTHER: Miscellaneous information
    """
    PERSON = "person"
    PLACE = "place"
    PREFERENCE = "preference"
    FACT = "fact"
    OTHER = "other"


class Memory(BaseModel):
    """
    Enhanced memory model with tiered memory support.
    
    Represents a memory entry that is stored for a user and can
    be synchronized between devices.
    
    Attributes:
        memory_id: Unique identifier
        user_id: ID of the user who owns this memory
        content: Memory content
        source: Source/type of the memory
        metadata: Additional metadata
        tags: Tags for categorization
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_synced: Whether memory is synced to cloud
        importance: Importance score (1-10)
        expiration: Optional expiration timestamp
    """
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    source: str = MemorySource.CORE.value
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_synced: bool = False
    importance: int = 5
    expiration: Optional[datetime] = None
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class CoreMemory(BaseModel):
    """
    Core memory model for long-term storage.
    
    Specialized model for core (long-term) memories that are particularly
    important for the user.
    
    Attributes:
        memory_id: Base memory identifier
        category: Memory category
        entity: Primary entity this memory relates to
        content: Memory content
        importance: Importance score (1-10)
        last_accessed: When this memory was last accessed
        created_at: Creation timestamp
    """
    memory_id: str
    category: MemoryCategory
    entity: Optional[str] = None
    content: str
    importance: int = 5
    last_accessed: Optional[datetime] = None
    created_at: datetime


class EventMemory(BaseModel):
    """
    Event memory model for time-based memories.
    
    Specialized model for event memories like reminders and appointments.
    
    Attributes:
        memory_id: Base memory identifier
        content: Event description
        event_time: When the event occurs
        expiration: When this memory expires
        completed: Whether the event is completed
        created_at: Creation timestamp
    """
    memory_id: str
    content: str
    event_time: datetime
    expiration: datetime
    completed: bool = False
    created_at: datetime


class ConversationMemory(BaseModel):
    """
    Conversation memory model for dialog context.
    
    Specialized model for conversation memories and summaries.
    
    Attributes:
        memory_id: Base memory identifier
        conversation_id: ID of the conversation
        content: Conversation content or summary
        entities: Key entities mentioned
        created_at: Creation timestamp
    """
    memory_id: str
    conversation_id: str
    content: str
    entities: List[str] = Field(default_factory=list)
    created_at: datetime


class SyncState(BaseModel):
    """
    Synchronization state model.
    
    Tracks the synchronization state between a device and the server.
    
    Attributes:
        user_id: User ID
        device_id: Device ID
        last_sync: Last synchronization timestamp
        synced_memory_ids: List of synced memory IDs
    """
    user_id: str
    device_id: str
    last_sync: Optional[datetime] = None
    synced_memory_ids: List[str] = Field(default_factory=list)
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class Conversation(BaseModel):
    """
    Conversation model.
    
    Represents a conversation between a user and the AI assistant.
    
    Attributes:
        conversation_id: Unique identifier
        user_id: ID of the user
        title: Conversation title
        messages: List of messages in the conversation
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
        summary: Optional conversation summary
    """
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Conversation"
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class MessageType(str, Enum):
    """
    Message type enumeration.
    
    Defines possible message types in conversations.
    
    Attributes:
        USER: Message from the user
        ASSISTANT: Message from the AI assistant
        SYSTEM: System message
        FUNCTION: Function call message
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class Message(BaseModel):
    """
    Message model.
    
    Represents a single message in a conversation.
    
    Attributes:
        message_id: Unique identifier
        conversation_id: ID of the conversation
        content: Message content
        type: Message type
        timestamp: Message timestamp
        metadata: Additional metadata
        token_count: Approximate token count of the message
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    content: str
    type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: Optional[int] = None
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class ContextItem(BaseModel):
    """
    Context item model.
    
    Represents an item in the context window.
    
    Attributes:
        content: Item content
        source: Source of the item
        importance: Importance score
        token_count: Token count
        created_at: Creation timestamp
    """
    content: str
    source: str  # "message", "memory", "system", etc.
    importance: float = 1.0
    token_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ApiKeyScope(str, Enum):
    """
    API key scope enumeration.
    
    Defines possible scopes for API keys.
    
    Attributes:
        READ: Read-only access
        WRITE: Read and write access
        ADMIN: Full administrative access
    """
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class ApiKey(BaseModel):
    """
    API key model.
    
    Represents an API key for programmatic access.
    
    Attributes:
        key_id: Unique identifier
        user_id: ID of the user who owns this key
        name: Key name
        prefix: Key prefix (for display)
        hashed_key: Hashed API key
        scopes: List of scopes
        created_at: Creation timestamp
        expires_at: Optional expiration timestamp
        last_used: Optional last usage timestamp
        is_active: Whether the key is active
    """
    key_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    name: str
    prefix: str
    hashed_key: str
    scopes: List[ApiKeyScope]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class RateLimitTier(str, Enum):
    """
    Rate limit tier enumeration.
    
    Defines possible rate limit tiers for users.
    
    Attributes:
        FREE: Free tier with basic limits
        STANDARD: Standard tier with higher limits
        PREMIUM: Premium tier with highest limits
    """
    FREE = "free"
    STANDARD = "standard"
    PREMIUM = "premium"


class RateLimitConfig(BaseModel):
    """
    Rate limit configuration model.
    
    Defines rate limits for different operations.
    
    Attributes:
        tier: Rate limit tier
        requests_per_minute: Requests allowed per minute
        requests_per_day: Requests allowed per day
        tokens_per_minute: Tokens allowed per minute
        tokens_per_day: Tokens allowed per day
    """
    tier: RateLimitTier
    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: int
    tokens_per_day: int
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class UserRateLimit(BaseModel):
    """
    User rate limit model.
    
    Tracks rate limit usage for a user.
    
    Attributes:
        user_id: User ID
        tier: Rate limit tier
        minute_count: Request count in current minute
        day_count: Request count in current day
        minute_tokens: Token count in current minute
        day_tokens: Token count in current day
        last_reset_minute: Last minute reset timestamp
        last_reset_day: Last day reset timestamp
    """
    user_id: str
    tier: RateLimitTier = RateLimitTier.FREE
    minute_count: int = 0
    day_count: int = 0
    minute_tokens: int = 0
    day_tokens: int = 0
    last_reset_minute: datetime = Field(default_factory=datetime.utcnow)
    last_reset_day: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


# Additional models for mental health support can be added in stage 3