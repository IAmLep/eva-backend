"""
Models module for EVA backend.

This module defines all data models used throughout the application,
providing type safety and validation with Pydantic.

Last updated: 2025-04-01 10:47:17
Version: v1.8.6
Created by: IAmLep
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


class Memory(BaseModel):
    """
    Memory model.
    
    Represents a memory entry that is stored for a user and can
    be synchronized between devices.
    
    Attributes:
        memory_id: Unique identifier
        user_id: ID of the user who owns this memory
        content: Memory content
        source: Source of the memory
        metadata: Additional metadata
        tags: Tags for categorization
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_synced: Whether memory is synced to cloud
    """
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    source: str = "user"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_synced: bool = False
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


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
    """
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    title: str = "New Conversation"
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
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
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    content: str
    type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class DeviceInfo(BaseModel):
    """
    Device information model.
    
    Tracks information about a user's device.
    
    Attributes:
        device_id: Unique device identifier
        user_id: User ID
        device_type: Type of device
        os_version: Operating system version
        app_version: Application version
        last_active: Last active timestamp
        push_token: Optional push notification token
    """
    device_id: str
    user_id: str
    device_type: str
    os_version: str
    app_version: str
    last_active: datetime = Field(default_factory=datetime.utcnow)
    push_token: Optional[str] = None
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class FunctionCall(BaseModel):
    """
    Function call model.
    
    Represents a function call made by the AI assistant.
    
    Attributes:
        function_name: Name of the function
        arguments: Function arguments
        result: Function result
        timestamp: Call timestamp
    """
    function_name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


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


class WebhookEvent(str, Enum):
    """
    Webhook event enumeration.
    
    Defines possible events that can trigger webhooks.
    
    Attributes:
        MESSAGE_CREATED: New message created
        MEMORY_CREATED: New memory created
        USER_CREATED: New user created
        RATE_LIMIT_EXCEEDED: Rate limit exceeded
    """
    MESSAGE_CREATED = "message.created"
    MEMORY_CREATED = "memory.created"
    USER_CREATED = "user.created"
    RATE_LIMIT_EXCEEDED = "rate_limit.exceeded"


class Webhook(BaseModel):
    """
    Webhook model.
    
    Represents a webhook configuration.
    
    Attributes:
        webhook_id: Unique identifier
        user_id: ID of the user who owns this webhook
        url: Webhook URL
        events: List of events to trigger this webhook
        is_active: Whether the webhook is active
        secret: Secret for webhook signature
        created_at: Creation timestamp
        metadata: Additional metadata
    """
    webhook_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    url: str
    events: List[WebhookEvent]
    is_active: bool = True
    secret: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True


class WebhookDelivery(BaseModel):
    """
    Webhook delivery model.
    
    Tracks webhook delivery attempts.
    
    Attributes:
        delivery_id: Unique identifier
        webhook_id: ID of the webhook
        event: Event type
        payload: Event payload
        status: Delivery status
        status_code: HTTP status code
        response: Response body
        delivered_at: Delivery timestamp
        retry_count: Number of retry attempts
    """
    delivery_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    webhook_id: str
    event: WebhookEvent
    payload: Dict[str, Any]
    status: str
    status_code: Optional[int] = None
    response: Optional[str] = None
    delivered_at: datetime = Field(default_factory=datetime.utcnow)
    retry_count: int = 0
    
    class Config:
        """Pydantic model configuration."""
        from_attributes = True