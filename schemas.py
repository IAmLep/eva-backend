# schemas.py
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Literal
import re

class ChatMessageBase(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)  #Basic input validation

class ChatMessageCreate(ChatMessageBase):
    pass

class ChatMessageResponse(ChatMessageBase):
    id: int
    timestamp: int

    class Config:
        from_attributes = True

class ConversationSummaryBase(BaseModel):
    label: str = Field(..., min_length=1, max_length=256)
    summary: str = Field(..., min_length=1, max_length=4096)

class ConversationSummaryCreate(ConversationSummaryBase):
    pass

class ConversationSummaryResponse(ConversationSummaryBase):
    id: int
    timestamp: int

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096)
    mode: Literal["text"] = "text"  # Changed from const to Literal
    device_id: Optional[str] = None  # Changed to Optional


class ChatResponse(BaseModel):
    answer: str
    memory_updated: Optional[bool] = False


class NoteBase(BaseModel):
    title: str = Field(..., max_length=50, min_length=1)
    content: str = Field(..., max_length=1000, min_length=1)

class NoteCreate(NoteBase):
    pass

class NoteResponse(NoteBase):
    id: int
    timestamp: int

    class Config:
        from_attributes = True


class WebSocketMessage(BaseModel):
    text: str


class AuthRequest(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=256)

class DeviceChange(BaseModel):
    """Changes made on a device to be synced to server"""
    entity_type: str  # "conversation", "message", "setting", etc.
    entity_id: str
    action: str  # "create", "update", "delete"
    data: Dict[str, Any]
    timestamp: str

class SyncRequest(BaseModel):
    """Request for device synchronization"""
    device_id: str
    last_sync_timestamp: Optional[str] = None
    local_changes: List[DeviceChange] = []

class SyncResponse(BaseModel):
    """Response containing data to be synced to device"""
    conversations: List[Dict[str, Any]] = []
    settings: Dict[str, Any] = {}
    server_timestamp: str