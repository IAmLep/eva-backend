"""
Schemas module for EVA backend.

This module defines the schema models used for API requests and responses,
providing validation and serialization with Pydantic.

Last updated: 2025-04-01 11:11:59
Version: v1.8.6
Created by: IAmLep
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, EmailStr, Field, validator, root_validator


class TokenRequest(BaseModel):
    """
    Request model for authentication token.
    
    Attributes:
        username: Username for authentication
        password: Password for authentication
    """
    username: str
    password: str


class Token(BaseModel):
    """
    Authentication token response model.
    
    Attributes:
        access_token: JWT token for authentication
        token_type: Type of token (always 'bearer')
        expires_at: Token expiration timestamp
    """
    access_token: str
    token_type: str
    expires_at: datetime


class UserBase(BaseModel):
    """
    Base user model with common fields.
    
    Attributes:
        username: Unique username
        email: Email address
        full_name: Optional full name
    """
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    
    @validator('username')
    def username_alphanumeric(cls, v):
        """Validate username contains only alphanumeric characters."""
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v


class UserCreate(UserBase):
    """
    User creation request model.
    
    Attributes:
        password: Password (will be hashed)
    """
    password: str
    
    @validator('password')
    def password_strong(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain an uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain a lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain a digit')
        return v


class UserUpdate(BaseModel):
    """
    User update request model.
    
    Attributes:
        email: Optional new email
        full_name: Optional new full name
        password: Optional new password
        preferences: Optional new preferences
    """
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    
    @validator('password')
    def password_strong(cls, v):
        """Validate password strength if provided."""
        if v is not None:
            if len(v) < 8:
                raise ValueError('Password must be at least 8 characters')
            if not any(c.isupper() for c in v):
                raise ValueError('Password must contain an uppercase letter')
            if not any(c.islower() for c in v):
                raise ValueError('Password must contain a lowercase letter')
            if not any(c.isdigit() for c in v):
                raise ValueError('Password must contain a digit')
        return v


class UserResponse(UserBase):
    """
    User response model.
    
    Attributes:
        id: User ID
        created_at: Account creation timestamp
        updated_at: Last update timestamp
        is_active: Whether user is active
        preferences: User preferences
    """
    id: str
    created_at: datetime
    updated_at: datetime
    is_active: bool
    preferences: Dict[str, Any] = Field(default_factory=dict)


class MemoryBase(BaseModel):
    """
    Base memory model with common fields.
    
    Attributes:
        content: Memory content
        source: Source of memory
        metadata: Additional metadata
        tags: Tags for categorization
    """
    content: str
    source: str = "user"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    
    @validator('content')
    def content_not_empty(cls, v):
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class MemoryCreate(MemoryBase):
    """
    Memory creation request model.
    """
    pass


class MemoryUpdate(BaseModel):
    """
    Memory update request model.
    
    Attributes:
        content: Optional new content
        metadata: Optional new metadata
        tags: Optional new tags
    """
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    @validator('content')
    def content_not_empty(cls, v):
        """Validate content is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip() if v is not None else v


class MemoryResponse(MemoryBase):
    """
    Memory response model.
    
    Attributes:
        memory_id: Unique memory identifier
        user_id: ID of user who owns the memory
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_synced: Whether memory is synced to cloud
    """
    memory_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    is_synced: bool


class ConversationStatus(str, Enum):
    """
    Conversation status enumeration.
    
    Attributes:
        ACTIVE: Active conversation
        ARCHIVED: Archived conversation
        DELETED: Soft-deleted conversation
    """
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ConversationCreate(BaseModel):
    """
    Conversation creation request model.
    
    Attributes:
        title: Conversation title
        metadata: Optional metadata
    """
    title: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationUpdate(BaseModel):
    """
    Conversation update request model.
    
    Attributes:
        title: Optional new title
        status: Optional new status
        metadata: Optional new metadata
    """
    title: Optional[str] = None
    status: Optional[ConversationStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class ConversationResponse(BaseModel):
    """
    Conversation response model.
    
    Attributes:
        conversation_id: Unique conversation identifier
        user_id: ID of user who owns the conversation
        title: Conversation title
        status: Conversation status
        message_count: Number of messages
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metadata: Additional metadata
    """
    conversation_id: str
    user_id: str
    title: str
    status: ConversationStatus
    message_count: int
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageRole(str, Enum):
    """
    Message role enumeration.
    
    Attributes:
        USER: Message from user
        ASSISTANT: Message from AI assistant
        SYSTEM: System message
        FUNCTION: Function call or result
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


class MessageCreate(BaseModel):
    """
    Message creation request model.
    
    Attributes:
        conversation_id: ID of conversation
        content: Message content
        role: Message role
        metadata: Optional metadata
    """
    conversation_id: str
    content: str
    role: MessageRole
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def content_not_empty(cls, v):
        """Validate content is not empty."""
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()


class MessageResponse(BaseModel):
    """
    Message response model.
    
    Attributes:
        message_id: Unique message identifier
        conversation_id: ID of conversation
        content: Message content
        role: Message role
        timestamp: Message timestamp
        metadata: Additional metadata
    """
    message_id: str
    conversation_id: str
    content: str
    role: MessageRole
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FunctionCallRequest(BaseModel):
    """
    Function call request model.
    
    Attributes:
        name: Function name
        arguments: Function arguments
    """
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class FunctionCallResponse(BaseModel):
    """
    Function call response model.
    
    Attributes:
        name: Function name
        arguments: Function arguments
        result: Function result
        error: Optional error message
        status: Call status
    """
    name: str
    arguments: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str


class ApiKeyCreate(BaseModel):
    """
    API key creation request model.
    
    Attributes:
        name: Key name
        scopes: Key scopes
        expires_in_days: Optional expiration in days
    """
    name: str
    scopes: List[str]
    expires_in_days: Optional[int] = None
    
    @validator('name')
    def name_not_empty(cls, v):
        """Validate name is not empty."""
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
    
    @validator('scopes')
    def scopes_valid(cls, v):
        """Validate scopes are valid."""
        valid_scopes = {"read", "write", "admin"}
        for scope in v:
            if scope not in valid_scopes:
                raise ValueError(f"Invalid scope: {scope}")
        return v
    
    @validator('expires_in_days')
    def expires_in_days_positive(cls, v):
        """Validate expires_in_days is positive if provided."""
        if v is not None and v <= 0:
            raise ValueError('expires_in_days must be positive')
        return v


class ApiKeyResponse(BaseModel):
    """
    API key response model.
    
    Attributes:
        key_id: Key ID
        name: Key name
        prefix: Key prefix
        key: Full key (only included in creation response)
        scopes: Key scopes
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        last_used: Last usage timestamp
        is_active: Whether key is active
    """
    key_id: str
    name: str
    prefix: str
    key: Optional[str] = None
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool


class WebhookCreate(BaseModel):
    """
    Webhook creation request model.
    
    Attributes:
        url: Webhook URL
        events: Events to trigger webhook
        description: Optional description
    """
    url: str
    events: List[str]
    description: Optional[str] = None
    
    @validator('url')
    def url_valid(cls, v):
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('events')
    def events_valid(cls, v):
        """Validate events are valid."""
        valid_events = {
            "message.created", "memory.created", "user.created", "rate_limit.exceeded"
        }
        for event in v:
            if event not in valid_events:
                raise ValueError(f"Invalid event: {event}")
        return v


class WebhookResponse(BaseModel):
    """
    Webhook response model.
    
    Attributes:
        webhook_id: Webhook ID
        url: Webhook URL
        events: Events triggering webhook
        description: Optional description
        is_active: Whether webhook is active
        created_at: Creation timestamp
        secret: Webhook secret (only included in creation response)
    """
    webhook_id: str
    url: str
    events: List[str]
    description: Optional[str] = None
    is_active: bool
    created_at: datetime
    secret: Optional[str] = None


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Attributes:
        detail: Error detail message
        code: Error code
        path: Request path
        field_errors: Optional field-specific errors
    """
    detail: str
    code: str
    path: str
    field_errors: Optional[Dict[str, List[str]]] = None


class SyncRequest(BaseModel):
    """
    Synchronization request model.
    
    Attributes:
        device_id: Device identifier
        last_sync: Last synchronization timestamp
        memories: Memories to synchronize
        deleted_memory_ids: IDs of deleted memories
    """
    device_id: str
    last_sync: Optional[datetime] = None
    memories: List[MemoryBase] = Field(default_factory=list)
    deleted_memory_ids: List[str] = Field(default_factory=list)


class SyncResponse(BaseModel):
    """
    Synchronization response model.
    
    Attributes:
        last_sync: Current synchronization timestamp
        server_memories: Memories from server
        conflict_memory_ids: IDs of memories with conflicts
        success: Whether sync was successful
    """
    last_sync: datetime
    server_memories: List[MemoryResponse]
    conflict_memory_ids: List[str] = Field(default_factory=list)
    success: bool


class TextGenerationRequest(BaseModel):
    """
    Text generation request model.
    
    Attributes:
        prompt: Generation prompt
        max_tokens: Maximum tokens to generate
        temperature: Generation temperature
        include_memory: Whether to include memories
    """
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.7
    include_memory: bool = True
    
    @validator('prompt')
    def prompt_not_empty(cls, v):
        """Validate prompt is not empty."""
        if not v.strip():
            raise ValueError('Prompt cannot be empty')
        return v.strip()
    
    @validator('max_tokens')
    def max_tokens_range(cls, v):
        """Validate max_tokens is in valid range."""
        if v < 1 or v > 4096:
            raise ValueError('max_tokens must be between 1 and 4096')
        return v
    
    @validator('temperature')
    def temperature_range(cls, v):
        """Validate temperature is in valid range."""
        if v < 0.0 or v > 1.0:
            raise ValueError('temperature must be between 0.0 and 1.0')
        return v


class TextGenerationResponse(BaseModel):
    """
    Text generation response model.
    
    Attributes:
        generated_text: Generated text
        prompt_tokens: Number of tokens in prompt
        completion_tokens: Number of tokens in completion
        total_tokens: Total tokens used
        memories_used: Whether memories were used
    """
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    memories_used: bool