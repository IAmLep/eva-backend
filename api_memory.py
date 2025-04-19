"""
Memory API endpoints for EVA backend.

This module provides API endpoints for managing the memory system,
including creating, retrieving, updating, deleting, and extracting memories.
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Path, Body, status
from pydantic import BaseModel, Field

from auth import get_current_user
# Corrected import from memory_manager
from memory_manager import get_memory_manager, MemoryCategory
from memory_extractor import get_memory_extractor, MemoryCommand # Import MemoryCommand if extractor returns it
# Corrected import from models - Removed CoreMemory, EventMemory
from models import Memory, User, MemorySource # Import MemorySource
from config import get_settings

# Logger configuration
logger = logging.getLogger(__name__)

# Router configuration
router = APIRouter(
    prefix="/memory",
    tags=["memory"],
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not found"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"}
    }
)


# --- Request and Response Models ---

class CreateCoreMemoryRequest(BaseModel):
    """Request model for creating core memories."""
    content: str = Field(..., min_length=1, max_length=2000)
    category: MemoryCategory
    entity: Optional[str] = None
    importance: int = Field(5, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = None


class CreateEventMemoryRequest(BaseModel):
    """Request model for creating event memories."""
    content: str = Field(..., min_length=1, max_length=2000)
    event_time: datetime
    expiration: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    """Response model for memory operations. Uses the base Memory model structure."""
    memory_id: str
    content: str
    source: MemorySource
    metadata: Dict[str, Any]
    tags: List[str]
    importance: int # Added importance to response
    created_at: datetime
    updated_at: datetime
    expiration: Optional[datetime] = None # Added expiration to response

    class Config:
        from_attributes = True


class MemoryListResponse(BaseModel):
    """Response model for memory list operations."""
    memories: List[MemoryResponse]
    count: int
    has_more: bool


class CreateMemoryResponse(BaseModel):
    """Response model for memory creation."""
    memory_id: str
    success: bool


class UpdateMemoryRequest(BaseModel):
    """Request model for updating memories. Allows partial updates."""
    content: Optional[str] = Field(None, min_length=1, max_length=2000)
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    importance: Optional[int] = Field(None, ge=1, le=10)


class ExtractMemoryResponse(BaseModel):
    """Response model for memory extraction from text."""
    command_found: bool = False
    command_type: Optional[str] = None
    command_content: Optional[str] = None
    command_category: Optional[MemoryCategory] = None
    command_entity: Optional[str] = None
    command_event_time: Optional[datetime] = None
    potential_memory: bool = False
    potential_content: Optional[str] = None
    potential_category: Optional[MemoryCategory] = None
    potential_importance: Optional[int] = None


# --- API Endpoints ---

@router.post("/core", response_model=CreateMemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_core_memory(
    request: CreateCoreMemoryRequest,
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager)
):
    """Create a new core memory (source = CORE)."""
    try:
        final_metadata = request.metadata or {}
        final_metadata['category'] = request.category.value
        if request.entity:
            final_metadata['entity'] = request.entity

        memory = await memory_manager.add_memory(
            user_id=user.id,
            content=request.content,
            source=MemorySource.CORE,
            importance=request.importance,
            metadata=final_metadata
        )
        logger.info(f"Created core memory {memory.memory_id} for user {user.id}")
        return CreateMemoryResponse(memory_id=memory.memory_id, success=True)
    except Exception as e:
        logger.exception(f"Error creating core memory for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create core memory.")


@router.post("/event", response_model=CreateMemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_event_memory(
    request: CreateEventMemoryRequest,
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager)
):
    """Create a new event memory (source = EVENT)."""
    try:
        final_metadata = request.metadata or {}
        final_metadata['event_time'] = request.event_time.isoformat()
        if request.expiration:
            final_metadata['expiration'] = request.expiration.isoformat()

        memory = await memory_manager.add_memory(
            user_id=user.id,
            content=request.content,
            source=MemorySource.EVENT,
            importance=5, # Default importance for events
            metadata=final_metadata,
            expiration=request.expiration
        )
        logger.info(f"Created event memory {memory.memory_id} for user {user.id}")
        return CreateMemoryResponse(memory_id=memory.memory_id, success=True)
    except Exception as e:
        logger.exception(f"Error creating event memory for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create event memory.")


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str = Path(..., title="Memory ID", description="The unique ID of the memory to retrieve."),
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager)
):
    """Get a specific memory by its ID."""
    try:
        memory = await memory_manager.get_memory(user_id=user.id, memory_id=memory_id)
        if not memory:
            logger.warning(f"Memory {memory_id} not found for user {user.id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found.")
        return MemoryResponse.model_validate(memory)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error retrieving memory {memory_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve memory.")


@router.get("/query/", response_model=MemoryListResponse)
async def query_memories(
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager),
    source: Optional[MemorySource] = Query(None, description="Filter by memory source (core, event, etc.)."),
    category: Optional[MemoryCategory] = Query(None, description="Filter core memories by category."),
    entity: Optional[str] = Query(None, description="Filter core memories by associated entity."),
    query: Optional[str] = Query(None, description="Perform semantic search using this query text."),
    include_past_events: bool = Query(False, description="For event source, include events that have passed."),
    days_ahead: int = Query(7, ge=0, description="For event source, look this many days ahead (0 means today only)."),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of memories to return."),
    offset: int = Query(0, ge=0, description="Number of memories to skip for pagination.")
):
    """Query memories based on various filters and semantic search."""
    try:
        search_params = {
            "user_id": user.id, "source": source, "category": category, "entity": entity,
            "query": query, "include_past_events": include_past_events, "days_ahead": days_ahead,
            "limit": limit + 1, "offset": offset
        }
        memories = await memory_manager.search_memories(**search_params)
        has_more = len(memories) > limit
        if has_more:
            memories = memories[:limit]
        memory_responses = [MemoryResponse.model_validate(mem) for mem in memories]
        return MemoryListResponse(memories=memory_responses, count=len(memory_responses), has_more=has_more)
    except Exception as e:
        logger.exception(f"Error querying memories for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to query memories.")


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    request: UpdateMemoryRequest,
    memory_id: str = Path(..., title="Memory ID", description="The unique ID of the memory to update."),
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager)
):
    """Update an existing memory. Allows partial updates to content, metadata, tags, importance."""
    try:
        updates = request.model_dump(exclude_unset=True)
        if not updates:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No update fields provided.")

        updated_memory = await memory_manager.update_memory(user_id=user.id, memory_id=memory_id, updates=updates)
        if not updated_memory:
            logger.warning(f"Update failed for memory {memory_id}, user {user.id}. Memory not found or ownership issue.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found or update failed.")

        logger.info(f"Updated memory {memory_id} for user {user.id}")
        return MemoryResponse.model_validate(updated_memory)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error updating memory {memory_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update memory.")


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str = Path(..., title="Memory ID", description="The unique ID of the memory to delete."),
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager)
):
    """Delete a specific memory by its ID."""
    try:
        deleted = await memory_manager.delete_memory(user_id=user.id, memory_id=memory_id)
        if not deleted:
            logger.warning(f"Delete failed for memory {memory_id}, user {user.id}. Memory not found or ownership issue.")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found or delete failed.")
        logger.info(f"Deleted memory {memory_id} for user {user.id}")
        return None # Return None for 204 status code
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting memory {memory_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete memory.")


@router.post("/event/{memory_id}/complete", status_code=status.HTTP_200_OK)
async def complete_event(
    memory_id: str = Path(..., title="Event Memory ID", description="The unique ID of the event memory to mark as complete."),
    user: User = Depends(get_current_user),
    memory_manager = Depends(get_memory_manager)
):
    """Mark an event memory as completed by updating its metadata."""
    try:
        success = await memory_manager.complete_event(user_id=user.id, memory_id=memory_id)
        if not success:
             logger.warning(f"Failed to mark event {memory_id} as complete for user {user.id}. Not found, not an event, or ownership issue.")
             raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event memory not found or could not be marked as complete.")
        logger.info(f"Marked event {memory_id} as completed for user {user.id}")
        return {"success": True, "message": f"Event {memory_id} marked as completed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error completing event {memory_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to complete event.")


# --- COMPLETED FUNCTION ---
@router.post("/text/extract", status_code=status.HTTP_200_OK, response_model=ExtractMemoryResponse)
async def extract_memory_from_text(
    text_input: str = Body(..., embed=True, alias="text", description="The text to analyze for potential memories or commands."),
    user: User = Depends(get_current_user), # User might be needed if extraction logic is user-specific
    memory_extractor = Depends(get_memory_extractor) # Inject dependency
):
    """
    Analyze text to extract explicit memory commands (e.g., "remember", "forget")
    or identify potentially useful information to store as a memory.

    Args:
        text_input: The text provided by the user.
        user: Authenticated user.
        memory_extractor: Injected MemoryExtractor instance.

    Returns:
        An ExtractMemoryResponse object detailing what was found (command or potential memory).
    """
    try:
        logger.debug(f"Extracting memory from text for user {user.id}. Text: '{text_input[:100]}...'")

        # Attempt to extract an explicit command first
        command: Optional[MemoryCommand] = await memory_extractor.extract_memory_command(text_input)

        if command:
            logger.info(f"Extracted command '{command.command_type}' for user {user.id}")
            return ExtractMemoryResponse(
                command_found=True,
                command_type=command.command_type,
                command_content=command.content,
                command_category=command.category,
                command_entity=command.entity,
                command_event_time=command.event_time
            )

        # If no command, check for potentially storable memory fragments
        # Assuming identify_potential_memory returns a dict or None
        potential_memory_info: Optional[Dict] = await memory_extractor.identify_potential_memory(text_input)

        if potential_memory_info:
             logger.info(f"Identified potential memory for user {user.id}")
             return ExtractMemoryResponse(
                  potential_memory=True,
                  potential_content=potential_memory_info.get("content"),
                  potential_category=potential_memory_info.get("category"),
                  potential_importance=potential_memory_info.get("importance")
             )

        # If neither command nor potential memory found
        logger.info(f"No command or potential memory found in text for user {user.id}")
        return ExtractMemoryResponse(command_found=False, potential_memory=False)

    except Exception as e:
        logger.exception(f"Error extracting memory from text for user {user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process text for memory extraction."
        )
# --- END OF FILE ---