"""
Memory Extractor module for EVA backend.

This module provides functionality for extracting, processing,
and managing memory data for both local storage and Firestore sync.


Version 3 working
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import get_current_active_user
from config import get_settings
from database import get_db_manager
from exceptions import DatabaseError, NotFoundException
from firestore_manager import get_firestore_client
from models import Memory, User
from cache_manager import cached

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)


class MemoryCreateRequest(BaseModel):
    """
    Memory creation request model.
    
    Attributes:
        content: Memory content
        source: Source of the memory
        metadata: Optional metadata for the memory
        tags: Optional tags for categorization
    """
    content: str
    source: str = "user"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    """
    Memory update request model.
    
    Attributes:
        content: Optional updated content
        metadata: Optional updated metadata
        tags: Optional updated tags
    """
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class MemoryResponse(BaseModel):
    """
    Memory response model.
    
    Attributes:
        memory_id: Unique identifier for the memory
        user_id: User ID who owns the memory
        content: Memory content
        source: Source of the memory
        metadata: Additional metadata
        tags: Tags for categorization
        created_at: Creation timestamp
        updated_at: Last update timestamp
        is_synced: Whether memory is synced to Firestore
    """
    memory_id: str
    user_id: str
    content: str
    source: str
    metadata: Dict[str, Any]
    tags: List[str]
    created_at: datetime
    updated_at: datetime
    is_synced: bool


class MemoryBulkOperationResponse(BaseModel):
    """
    Response model for bulk memory operations.
    
    Attributes:
        success_count: Number of successfully processed items
        failed_count: Number of failed items
        failed_ids: List of IDs that failed processing
    """
    success_count: int
    failed_count: int
    failed_ids: List[str] = Field(default_factory=list)


class MemorySyncRequest(BaseModel):
    """
    Memory synchronization request model.
    
    Attributes:
        device_id: Device identifier
        memories: List of memories to sync
        last_sync: Last synchronization timestamp
    """
    device_id: str
    memories: List[Memory]
    last_sync: Optional[datetime] = None


class MemorySyncResponse(BaseModel):
    """
    Memory synchronization response model.
    
    Attributes:
        synced_count: Number of memories synced
        conflict_count: Number of conflicts resolved
        deleted_count: Number of memories deleted
        last_sync: New synchronization timestamp
        synced_memory_ids: List of synced memory IDs
    """
    synced_count: int
    conflict_count: int
    deleted_count: int
    last_sync: datetime
    synced_memory_ids: List[str]


async def extract_memory_from_text(
    text: str, 
    user_id: str, 
    metadata: Optional[Dict[str, Any]] = None
) -> Memory:
    """
    Extract and create a memory from text content.
    
    Args:
        text: Text content to create memory from
        user_id: User ID to associate with the memory
        metadata: Optional metadata for the memory
        
    Returns:
        Memory: Created memory object
        
    Raises:
        DatabaseError: If memory creation fails
    """
    try:
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Create memory object
        memory = Memory(
            memory_id=memory_id,
            user_id=user_id,
            content=text,
            source="extraction",
            metadata={
                **metadata,
                "extracted_at": now.isoformat()
            },
            tags=["extracted"],
            created_at=now,
            updated_at=now,
            is_synced=False
        )
        
        # Store in database
        db = get_db_manager()
        await db.create_memory(memory)
        
        logger.info(f"Extracted memory {memory_id} for user {user_id}")
        return memory
    
    except Exception as e:
        logger.error(f"Memory extraction error: {str(e)}")
        raise DatabaseError(f"Failed to extract memory: {str(e)}")


@cached(ttl=60, key_prefix="relevant_memories")
async def find_relevant_memories(
    user_id: str, 
    query: str, 
    limit: int = 5
) -> List[Memory]:
    """
    Find memories relevant to a query.
    
    Args:
        user_id: User ID to get memories for
        query: Query to match against memories
        limit: Maximum number of memories to return
        
    Returns:
        List[Memory]: List of relevant memories
        
    Raises:
        DatabaseError: If query fails
    """
    try:
        db = get_db_manager()
        
        # In a real implementation, this would use semantic search
        # For now, we'll use a simple keyword matching approach
        query_terms = query.lower().split()
        
        # Get all memories for user
        all_memories = await db.get_user_memories(user_id)
        
        # Score memories by relevance
        scored_memories = []
        for memory in all_memories:
            score = 0
            content_lower = memory.content.lower()
            
            # Simple term frequency scoring
            for term in query_terms:
                if term in content_lower:
                    score += content_lower.count(term)
            
            # Also consider recency
            days_old = (datetime.utcnow() - memory.created_at).days
            recency_score = max(0, 30 - days_old) / 30  # Favor last 30 days
            
            # Combine scores
            final_score = score + (recency_score * 2)
            
            if score > 0:  # Only include if there's a match
                scored_memories.append((memory, final_score))
        
        # Sort by score and take top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [memory for memory, _ in scored_memories[:limit]]
        
        logger.info(f"Found {len(top_memories)} relevant memories for user {user_id}")
        return top_memories
    
    except Exception as e:
        logger.error(f"Error finding relevant memories: {str(e)}")
        raise DatabaseError(f"Failed to find relevant memories: {str(e)}")


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_data: MemoryCreateRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Create a new memory.
    
    Args:
        memory_data: Memory creation request
        current_user: Current authenticated user
        
    Returns:
        Dict: Created memory data
        
    Raises:
        HTTPException: If memory creation fails
    """
    try:
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Create memory object
        memory = Memory(
            memory_id=memory_id,
            user_id=current_user.id,
            content=memory_data.content,
            source=memory_data.source,
            metadata=memory_data.metadata,
            tags=memory_data.tags,
            created_at=now,
            updated_at=now,
            is_synced=False
        )
        
        # Store in database
        db = get_db_manager()
        await db.create_memory(memory)
        
        logger.info(f"Created memory {memory_id} for user {current_user.id}")
        
        # Return created memory
        return memory.model_dump()
    
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory: {str(e)}"
        )


@router.get("", response_model=List[MemoryResponse])
async def get_memories(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    tags: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user)
) -> List[Dict]:
    """
    Get user memories with filtering options.
    
    Args:
        limit: Maximum number of memories to return
        offset: Offset for pagination
        tags: Comma-separated list of tags to filter by
        start_date: Start date for filtering
        end_date: End date for filtering
        current_user: Current authenticated user
        
    Returns:
        List[Dict]: List of memory data
        
    Raises:
        HTTPException: If retrieval fails
    """
    try:
        # Parse tags if provided
        tag_list = tags.split(",") if tags else None
        
        # Get database manager
        db = get_db_manager()
        
        # Get memories with filters
        memories = await db.get_user_memories(
            current_user.id,
            limit=limit,
            offset=offset,
            tags=tag_list,
            start_date=start_date,
            end_date=end_date
        )
        
        logger.info(f"Retrieved {len(memories)} memories for user {current_user.id}")
        
        # Return memories
        return [memory.model_dump() for memory in memories]
    
    except Exception as e:
        logger.error(f"Error retrieving memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}"
        )


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Get a specific memory by ID.
    
    Args:
        memory_id: Memory ID to retrieve
        current_user: Current authenticated user
        
    Returns:
        Dict: Memory data
        
    Raises:
        HTTPException: If memory not found or retrieval fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get memory
        memory = await db.get_memory(memory_id)
        
        if not memory:
            logger.warning(f"Memory {memory_id} not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        
        # Check ownership
        if memory.user_id != current_user.id:
            logger.warning(f"User {current_user.id} attempted to access memory {memory_id} owned by {memory.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this memory"
            )
        
        logger.info(f"Retrieved memory {memory_id} for user {current_user.id}")
        
        # Return memory
        return memory.model_dump()
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memory: {str(e)}"
        )


@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    memory_data: MemoryUpdateRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Update a memory.
    
    Args:
        memory_id: Memory ID to update
        memory_data: Memory update data
        current_user: Current authenticated user
        
    Returns:
        Dict: Updated memory data
        
    Raises:
        HTTPException: If update fails or memory not found
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Get existing memory
        memory = await db.get_memory(memory_id)
        
        if not memory:
            logger.warning(f"Memory {memory_id} not found for update")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found"
            )
        
        # Check ownership
        if memory.user_id != current_user.id:
            logger.warning(f"User {current_user.id} attempted to update memory {memory_id} owned by {memory.user_id}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this memory"
            )
        
        # Update fields if provided
        update_data = {}
        if memory_data.content is not None:
            update_data["content"] = memory_data.content
        
        if memory_data.metadata is not None:
            update_data["metadata"] = memory_data.metadata
        
        if memory_data.tags is not None:
            update_data["tags"] = memory_data.tags
        
        # Set updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        update_data["is_synced"] = False  # Mark as needing sync
        
        # Update memory
        success = await db.update_memory(memory_id, update_data)
        
        if not success:
            logger.error(f"Failed to update memory {memory_id}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update memory"
            )
        
        # Get updated memory
        updated_memory = await db.get_memory(memory_id)
        
        logger.info(f"Updated memory {memory_id} for user {current_user.id}")
        
        # Return updated memory
        return updated_memory.model_dump()
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error updating memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update memory: {str(e)}"
        )


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Delete a memory.
    
    Args:
        memory_id: Memory ID to delete
        current_user: Current authenticated user
        
    Returns:
        Dict: Deletion confirmation
        
    Raises:
        HTTPException: If deletion fails or memory not found
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Delete memory
        success = await db.delete_memory(current_user.id, memory_id)
        
        if not success:
            logger.warning(f"Memory {memory_id} not found for deletion")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Memory not found or already deleted"
            )
        
        logger.info(f"Deleted memory {memory_id} for user {current_user.id}")
        
        return {"success": True, "message": "Memory deleted successfully"}
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}"
        )


@router.post("/sync", response_model=MemorySyncResponse)
async def sync_memories(
    sync_request: MemorySyncRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Synchronize memories between device and server.
    
    Implements the offline-first approach, handling conflicts
    and tracking sync state.
    
    Args:
        sync_request: Memory synchronization request
        current_user: Current authenticated user
        
    Returns:
        Dict: Synchronization results
        
    Raises:
        HTTPException: If synchronization fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Validate that all memories belong to current user
        for memory in sync_request.memories:
            if memory.user_id != current_user.id:
                logger.warning(f"Sync attempt with memory owned by {memory.user_id} from user {current_user.id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot sync memories owned by other users"
                )
        
        # Perform sync
        merged_memories, sync_state = await db.sync_memories(
            current_user.id,
            sync_request.memories,
            sync_request.device_id
        )
        
        # Count conflicts (memories that were merged)
        synced_ids = set(m.memory_id for m in sync_request.memories)
        server_ids = set(m.memory_id for m in merged_memories if m.memory_id not in synced_ids)
        
        # Calculate counts
        synced_count = len(synced_ids)
        conflict_count = len([m for m in merged_memories if m.memory_id in synced_ids and m.memory_id in server_ids])
        
        # Get deleted memories since last sync
        deleted_count = 0
        if sync_request.last_sync:
            deleted_memories = await db.get_deleted_memories(
                current_user.id,
                sync_request.last_sync
            )
            deleted_count = len(deleted_memories)
        
        logger.info(f"Synced {synced_count} memories for user {current_user.id}, device {sync_request.device_id}")
        
        # Return sync results
        return {
            "synced_count": synced_count,
            "conflict_count": conflict_count,
            "deleted_count": deleted_count,
            "last_sync": datetime.utcnow(),
            "synced_memory_ids": list(synced_ids)
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error syncing memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to sync memories: {str(e)}"
        )


@router.post("/bulk-delete", response_model=MemoryBulkOperationResponse)
async def bulk_delete_memories(
    memory_ids: List[str],
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Delete multiple memories in bulk.
    
    Args:
        memory_ids: List of memory IDs to delete
        current_user: Current authenticated user
        
    Returns:
        Dict: Bulk operation results
        
    Raises:
        HTTPException: If bulk deletion fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Track results
        success_count = 0
        failed_ids = []
        
        # Process each memory
        for memory_id in memory_ids:
            try:
                success = await db.delete_memory(current_user.id, memory_id)
                if success:
                    success_count += 1
                else:
                    failed_ids.append(memory_id)
            except Exception:
                failed_ids.append(memory_id)
        
        logger.info(f"Bulk deleted {success_count}/{len(memory_ids)} memories for user {current_user.id}")
        
        # Return results
        return {
            "success_count": success_count,
            "failed_count": len(failed_ids),
            "failed_ids": failed_ids
        }
    
    except Exception as e:
        logger.error(f"Error in bulk memory deletion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform bulk deletion: {str(e)}"
        )


@router.post("/cleanup", response_model=Dict[str, int])
async def cleanup_old_memories(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Clean up old memories to prevent Firestore duplication.
    
    Args:
        days: Age threshold in days for memories to delete
        current_user: Current authenticated user
        
    Returns:
        Dict: Cleanup results with count of deleted memories
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        # Get database manager
        db = get_db_manager()
        
        # Perform cleanup
        count = await db.cleanup_old_memories(current_user.id, days)
        
        logger.info(f"Cleaned up {count} old memories for user {current_user.id}")
        
        # Return results
        return {"deleted_count": count}
    
    except Exception as e:
        logger.error(f"Error cleaning up old memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean up old memories: {str(e)}"
        )


@router.post("/remove-duplicates", response_model=Dict[str, int])
async def remove_duplicate_memories(
    current_user: User = Depends(get_current_active_user)
) -> Dict:
    """
    Remove duplicate memories for a user.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict: Results with count of removed duplicates
        
    Raises:
        HTTPException: If duplicate removal fails
    """
    try:
        # Get Firestore manager
        firestore = get_firestore_client()
        
        # Remove duplicates
        removed_count = await firestore.cleanup_duplicate_memories(current_user.id)
        
        logger.info(f"Removed {removed_count} duplicate memories for user {current_user.id}")
        
        # Return results
        return {"removed_count": removed_count}
    
    except Exception as e:
        logger.error(f"Error removing duplicate memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove duplicate memories: {str(e)}"
        )


@router.get("/search", response_model=List[MemoryResponse])
async def search_memories(
    query: str,
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
) -> List[Dict]:
    """
    Search for memories by content.
    
    Args:
        query: Search query
        limit: Maximum number of results
        current_user: Current authenticated user
        
    Returns:
        List[Dict]: List of matching memories
        
    Raises:
        HTTPException: If search fails
    """
    try:
        # Find relevant memories
        memories = await find_relevant_memories(
            current_user.id,
            query,
            limit
        )
        
        logger.info(f"Found {len(memories)} memories matching '{query}' for user {current_user.id}")
        
        # Return memories
        return [memory.model_dump() for memory in memories]
    
    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}"
        )