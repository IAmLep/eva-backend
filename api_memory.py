"""
Memory API endpoints for EVA backend.

This module provides API endpoints for managing the memory system,
including creating, retrieving, updating, and deleting memories.

Create this new file to add memory management endpoints.

Current Date: 2025-04-13 11:13:26
Current User: IAmLep
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field

from auth import get_current_user
from memory_manager import get_memory_manager, MemoryType, MemoryCategory
from memory_extractor import get_memory_extractor
from models import Memory, User, CoreMemory, EventMemory
from config import get_settings

# Logger configuration
logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    responses={404: {"description": "Not found"}}
)


# Request and response models
class CreateCoreMemoryRequest(BaseModel):
    """
    Request model for creating core memories.
    
    Attributes:
        content: Memory content
        category: Memory category
        entity: Optional entity this memory relates to
        importance: Importance score (1-10)
        metadata: Optional additional metadata
    """
    content: str = Field(..., min_length=1, max_length=2000)
    category: MemoryCategory
    entity: Optional[str] = None
    importance: int = Field(5, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None


class CreateEventMemoryRequest(BaseModel):
    """
    Request model for creating event memories.
    
    Attributes:
        content: Event description
        event_time: When the event occurs
        expiration: Optional expiration time
        metadata: Optional additional metadata
    """
    content: str = Field(..., min_length=1, max_length=2000)
    event_time: datetime
    expiration: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    """
    Response model for memory operations.
    
    Attributes:
        memory_id: Memory identifier
        content: Memory content
        source: Memory source/type
        metadata: Additional metadata
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    memory_id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime


class MemoryListResponse(BaseModel):
    """
    Response model for memory list operations.
    
    Attributes:
        memories: List of memories
        count: Total count
        has_more: Whether there are more results
    """
    memories: List[MemoryResponse]
    count: int
    has_more: bool


class CreateMemoryResponse(BaseModel):
    """
    Response model for memory creation.
    
    Attributes:
        memory_id: Created memory identifier
        success: Success status
    """
    memory_id: str
    success: bool


class UpdateMemoryRequest(BaseModel):
    """
    Request model for updating memories.
    
    Attributes:
        content: Optional new content
        metadata: Optional metadata updates
        tags: Optional tags updates
    """
    content: Optional[str] = Field(None, min_length=1, max_length=2000)
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


# API endpoints
@router.post("/core", response_model=CreateMemoryResponse)
async def create_core_memory(
    request: CreateCoreMemoryRequest,
    user: User = Depends(get_current_user)
):
    """
    Create a new core memory.
    
    Args:
        request: Memory creation request
        user: Authenticated user
        
    Returns:
        CreateMemoryResponse: Creation response
    """
    try:
        memory_manager = get_memory_manager()
        
        memory = await memory_manager.create_core_memory(
            user_id=user.id,
            content=request.content,
            category=request.category,
            entity=request.entity,
            importance=request.importance,
            metadata=request.metadata or {}
        )
        
        logger.info(f"Created core memory {memory.memory_id} for user {user.id}")
        
        return CreateMemoryResponse(
            memory_id=memory.memory_id,
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error creating core memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory: {str(e)}"
        )


@router.post("/event", response_model=CreateMemoryResponse)
async def create_event_memory(
    request: CreateEventMemoryRequest,
    user: User = Depends(get_current_user)
):
    """
    Create a new event memory.
    
    Args:
        request: Event memory creation request
        user: Authenticated user
        
    Returns:
        CreateMemoryResponse: Creation response
    """
    try:
        memory_manager = get_memory_manager()
        
        memory = await memory_manager.create_event_memory(
            user_id=user.id,
            content=request.content,
            event_time=request.event_time,
            expiration=request.expiration,
            metadata=request.metadata or {}
        )
        
        logger.info(f"Created event memory {memory.memory_id} for user {user.id}")
        
        return CreateMemoryResponse(
            memory_id=memory.memory_id,
            success=True
        )
    
    except Exception as e:
        logger.error(f"Error creating event memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create event memory: {str(e)}"
        )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str = Path(..., title="Memory ID"),
    user: User = Depends(get_current_user)
):
    """
    Get a specific memory.
    
    Args:
        memory_id: Memory ID to retrieve
        user: Authenticated user
        
    Returns:
        MemoryResponse: Memory data
        
    Raises:
        HTTPException: If memory not found or unauthorized
    """
    try:
        memory_manager = get_memory_manager()
        
        memory = await memory_manager.get_memory(memory_id, user.id)
        
        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found"
            )
        
        return MemoryResponse(
            memory_id=memory.memory_id,
            content=memory.content,
            source=memory.source,
            metadata=memory.metadata,
            tags=memory.tags,
            created_at=memory.created_at,
            updated_at=memory.updated_at
        )
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error retrieving memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}"
        )


@router.get("/core", response_model=MemoryListResponse)
async def get_core_memories(
    category: Optional[MemoryCategory] = None,
    entity: Optional[str] = None,
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user)
):
    """
    Get core memories, optionally filtered by category or entity.
    
    Args:
        category: Optional category filter
        entity: Optional entity filter
        limit: Maximum number of memories to return
        offset: Pagination offset
        user: Authenticated user
        
    Returns:
        MemoryListResponse: List of memories
    """
    try:
        memory_manager = get_memory_manager()
        
        memories = await memory_manager.get_core_memories(
            user_id=user.id,
            category=category,
            entity=entity,
            limit=limit + 1  # Request one extra to check for more
        )
        
        # Check if there are more results
        has_more = len(memories) > limit
        if has_more:
            memories = memories[:limit]  # Remove the extra one
        
        memory_responses = [
            MemoryResponse(
                memory_id=memory.memory_id,
                content=memory.content,
                source=memory.source,
                metadata=memory.metadata,
                tags=memory.tags,
                created_at=memory.created_at,
                updated_at=memory.updated_at
            )
            for memory in memories
        ]
        
        return MemoryListResponse(
            memories=memory_responses,
            count=len(memory_responses),
            has_more=has_more
        )
    
    except Exception as e:
        logger.error(f"Error retrieving core memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


@router.get("/event", response_model=MemoryListResponse)
async def get_event_memories(
    include_past: bool = Query(False),
    days_ahead: int = Query(7, ge=1, le=30),
    limit: int = Query(10, ge=1, le=100),
    user: User = Depends(get_current_user)
):
    """
    Get upcoming event memories.
    
    Args:
        include_past: Whether to include past events
        days_ahead: How many days ahead to look
        limit: Maximum number of events to return
        user: Authenticated user
        
    Returns:
        MemoryListResponse: List of event memories
    """
    try:
        memory_manager = get_memory_manager()
        
        events = await memory_manager.get_event_memories(
            user_id=user.id,
            include_past=include_past,
            days_ahead=days_ahead,
            limit=limit + 1  # Request one extra to check for more
        )
        
        # Check if there are more results
        has_more = len(events) > limit
        if has_more:
            events = events[:limit]  # Remove the extra one
        
        event_responses = [
            MemoryResponse(
                memory_id=event.memory_id,
                content=event.content,
                source=event.source,
                metadata=event.metadata,
                tags=event.tags,
                created_at=event.created_at,
                updated_at=event.updated_at
            )
            for event in events
        ]
        
        return MemoryListResponse(
            memories=event_responses,
            count=len(event_responses),
            has_more=has_more
        )
    
    except Exception as e:
        logger.error(f"Error retrieving event memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve events: {str(e)}"
        )


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    request: UpdateMemoryRequest,
    memory_id: str = Path(..., title="Memory ID"),
    user: User = Depends(get_current_user)
):
    """
    Update a memory.
    
    Args:
        request: Update request
        memory_id: Memory ID to update
        user: Authenticated user
        
    Returns:
        MemoryResponse: Updated memory
        
    Raises:
        HTTPException: If memory not found or unauthorized
    """
    try:
        memory_manager = get_memory_manager()
        
        # Verify memory exists and user owns it
        existing_memory = await memory_manager.get_memory(memory_id, user.id)
        
        if not existing_memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found"
            )
        
        # Prepare updates
        updates = {}
        
        if request.content is not None:
            updates["content"] = request.content
            
        if request.metadata is not None:
            updates["metadata"] = {**existing_memory.metadata, **request.metadata}
            
        if request.tags is not None:
            updates["tags"] = request.tags
        
        # Update memory
        success = await memory_manager.update_memory(memory_id, user.id, updates)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update memory"
            )
        
        # Get updated memory
        updated_memory = await memory_manager.get_memory(memory_id, user.id)
        
        logger.info(f"Updated memory {memory_id} for user {user.id}")
        
        return MemoryResponse(
            memory_id=updated_memory.memory_id,
            content=updated_memory.content,
            source=updated_memory.source,
            metadata=updated_memory.metadata,
            tags=updated_memory.tags,
            created_at=updated_memory.created_at,
            updated_at=updated_memory.updated_at
        )
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error updating memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update memory: {str(e)}"
        )


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str = Path(..., title="Memory ID"),
    user: User = Depends(get_current_user)
):
    """
    Delete a memory.
    
    Args:
        memory_id: Memory ID to delete
        user: Authenticated user
        
    Returns:
        dict: Success response
        
    Raises:
        HTTPException: If memory not found or unauthorized
    """
    try:
        memory_manager = get_memory_manager()
        
        success = await memory_manager.delete_memory(user.id, memory_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found"
            )
        
        logger.info(f"Deleted memory {memory_id} for user {user.id}")
        
        return {"success": True, "message": f"Memory {memory_id} deleted"}
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}"
        )


@router.post("/event/{memory_id}/complete")
async def complete_event(
    memory_id: str = Path(..., title="Memory ID"),
    user: User = Depends(get_current_user)
):
    """
    Mark an event memory as completed.
    
    Args:
        memory_id: Memory ID to update
        user: Authenticated user
        
    Returns:
        dict: Success response
        
    Raises:
        HTTPException: If memory not found or unauthorized
    """
    try:
        memory_manager = get_memory_manager()
        
        success = await memory_manager.complete_event(memory_id, user.id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event memory {memory_id} not found"
            )
        
        logger.info(f"Marked event {memory_id} as completed for user {user.id}")
        
        return {"success": True, "message": f"Event {memory_id} marked as completed"}
    
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"Error completing event {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete event: {str(e)}"
        )


@router.post("/text/extract")
async def extract_memory_from_text(
    text: str = Body(..., embed=True),
    user: User = Depends(get_current_user)
):
    """
    Extract memory commands from text.
    
    Args:
        text: Text to analyze
        user: Authenticated user
        
    Returns:
        dict: Extraction results
    """
    try:
        memory_extractor = get_memory_extractor()
        
        # Extract memory command
        command = await memory_extractor.extract_memory_command(text)
        
        if command:
            return {
                "command_found": True,
                "command_type": command.command_type,
                "content": command.content,
                "entity": command.entity,
                "category": command.category.value if command.category else None,
                "event_time": command.event_time.isoformat() if command.event_time else None
            }
        
        # Check for potential memory
        potential_memory = await memory_extractor.identify_potential_memory(text)
        
        if potential_memory:
            return {
                "command_found": False,
                "potential_memory": True,
                "content": potential_memory.get("content"),
                "category": potential_memory.get("category").value,
                "importance": potential_memory.get("importance")
            }
        
        return {
            "command_found": False,
            "potential_memory": False
        }
    
    except Exception as e:
        logger.error(f"Error extracting memory from text: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract memory: {str(e)}"
        )