"""
API Sync module for EVA backend.

This module provides synchronization functionality for the offline-first
approach between device databases and Firestore backup.
"""
"""
Version 3 working
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from auth import get_current_active_user
from config import get_settings
from database import get_db_manager
from exceptions import DatabaseError, SyncError
from firestore_manager import get_firestore_client
from models import Memory, SyncState, User

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)


class SyncRequest(BaseModel):
    """
    Synchronization request model.
    
    Attributes:
        device_id: Unique identifier for the device
        last_sync: Optional timestamp of last synchronization
        memories: List of memories to synchronize
        deleted_ids: Optional list of locally deleted memory IDs
    """
    device_id: str
    last_sync: Optional[datetime] = None
    memories: List[Memory]
    deleted_ids: List[str] = Field(default_factory=list)


class SyncConflict(BaseModel):
    """
    Synchronization conflict model.
    
    Attributes:
        memory_id: Memory ID with conflict
        device_version: Device version of the memory
        server_version: Server version of the memory
        resolution: How the conflict was resolved
    """
    memory_id: str
    device_version: datetime
    server_version: datetime
    resolution: str = "server_wins"  # Options: server_wins, device_wins, merged


class SyncResponse(BaseModel):
    """
    Synchronization response model.
    
    Attributes:
        sync_id: Unique identifier for this sync operation
        device_id: Device identifier from the request
        timestamp: Timestamp of synchronization
        added_count: Number of memories added to server
        updated_count: Number of memories updated on server
        deleted_count: Number of memories deleted
        conflicts: List of conflicts detected and resolved
        server_memories: List of memories from server
        last_sync: New last sync timestamp to use in next request
    """
    sync_id: str
    device_id: str
    timestamp: datetime
    added_count: int
    updated_count: int
    deleted_count: int
    conflicts: List[SyncConflict] = Field(default_factory=list)
    server_memories: List[Memory] = Field(default_factory=list)
    last_sync: datetime


class SyncStats(BaseModel):
    """
    Synchronization statistics model.
    
    Attributes:
        device_id: Device identifier
        last_sync: Last successful sync timestamp
        total_syncs: Total number of syncs performed
        total_memories: Total number of memories synced
        memory_count: Current number of memories
        sync_frequency: Average time between syncs (in hours)
    """
    device_id: str
    last_sync: Optional[datetime]
    total_syncs: int
    total_memories: int
    memory_count: int
    sync_frequency: Optional[float] = None


class CleanupRequest(BaseModel):
    """
    Cleanup request model.
    
    Attributes:
        days_threshold: Age threshold in days for memories to clean up
        device_id: Optional device ID to limit cleanup scope
        dry_run: Whether to simulate cleanup without deleting
    """
    days_threshold: int = 30
    device_id: Optional[str] = None
    dry_run: bool = False


class CleanupResponse(BaseModel):
    """
    Cleanup response model.
    
    Attributes:
        deleted_count: Number of memories deleted
        duplicate_count: Number of duplicates removed
        affected_devices: List of affected device IDs
        dry_run: Whether this was a simulation
    """
    deleted_count: int
    duplicate_count: int
    affected_devices: List[str]
    dry_run: bool


async def synchronize_memories(
    user_id: str,
    device_id: str,
    memories: List[Memory],
    deleted_ids: List[str] = None,
    last_sync: Optional[datetime] = None
) -> Tuple[SyncResponse, List[Memory]]:
    """
    Synchronize memories between device and server.
    
    Implements the offline-first approach with conflict resolution
    and tracking of sync state.
    
    Args:
        user_id: User ID owning the memories
        device_id: Device ID for sync tracking
        memories: List of memories from device
        deleted_ids: List of memory IDs deleted on device
        last_sync: Last synchronization timestamp
        
    Returns:
        Tuple[SyncResponse, List[Memory]]: 
            Sync response and merged memories list
            
    Raises:
        DatabaseError: If sync operation fails
        SyncError: If synchronization conflicts cannot be resolved
    """
    try:
        # Generate sync ID
        sync_id = str(uuid.uuid4())
        now = datetime.utcnow()
        deleted_ids = deleted_ids or []
        
        # Get database and Firestore clients
        db = get_db_manager()
        fs = get_firestore_client()
        
        # Get current sync state for device
        sync_ref = (fs.collection("sync_states")
                   .document(f"{user_id}_{device_id}"))
        sync_doc = sync_ref.get()
        
        # Initialize tracking variables
        added_count = 0
        updated_count = 0
        deleted_count = 0
        conflicts = []
        
        # Get or create sync state
        current_sync = None
        if sync_doc.exists:
            current_sync = SyncState(**sync_doc.to_dict())
        else:
            current_sync = SyncState(
                user_id=user_id,
                device_id=device_id,
                last_sync=None,
                synced_memory_ids=[]
            )
        
        # Get server memories for this user
        server_memories_query = (fs.collection("memories")
                                .where("user_id", "==", user_id))
        server_memories = [Memory(**doc.to_dict()) for doc in server_memories_query.stream()]
        
        # Index server memories by ID for quick lookup
        server_memory_map = {m.memory_id: m for m in server_memories}
        
        # Process device memories
        device_memory_map = {m.memory_id: m for m in memories}
        
        # Track processed memories
        processed_ids = set()
        synced_ids = set(current_sync.synced_memory_ids)
        final_memories = []
        
        # First, handle deletions from device
        for memory_id in deleted_ids:
            if memory_id in server_memory_map:
                # Delete from server
                await db.delete_memory(user_id, memory_id)
                deleted_count += 1
                
                # Remove from tracking sets
                if memory_id in synced_ids:
                    synced_ids.remove(memory_id)
                if memory_id in server_memory_map:
                    del server_memory_map[memory_id]
            
            processed_ids.add(memory_id)
        
        # Process device memories - new and updated
        for memory_id, device_memory in device_memory_map.items():
            # Skip already processed (deleted)
            if memory_id in processed_ids:
                continue
                
            # Verify memory belongs to user
            if device_memory.user_id != user_id:
                logger.warning(f"Sync attempt with memory owned by {device_memory.user_id} from user {user_id}")
                continue
            
            # Check if memory exists on server
            if memory_id in server_memory_map:
                server_memory = server_memory_map[memory_id]
                
                # Check for conflicts - both sides updated since last sync
                if (last_sync and 
                    server_memory.updated_at > last_sync and 
                    device_memory.updated_at > last_sync):
                    
                    # Resolve conflict - server wins by default
                    # In a more sophisticated implementation, you might merge fields
                    conflict = SyncConflict(
                        memory_id=memory_id,
                        device_version=device_memory.updated_at,
                        server_version=server_memory.updated_at,
                        resolution="server_wins"
                    )
                    conflicts.append(conflict)
                    
                    # Keep server version
                    final_memories.append(server_memory)
                
                # No conflict, just newer on device
                elif not last_sync or device_memory.updated_at > server_memory.updated_at:
                    # Update server with device version
                    device_memory.is_synced = True
                    await db.update_memory(memory_id, device_memory.model_dump())
                    updated_count += 1
                    
                    # Add to final memories
                    final_memories.append(device_memory)
                    
                else:
                    # Server version is newer or same, keep it
                    final_memories.append(server_memory)
            
            else:
                # New memory, add to server
                device_memory.is_synced = True
                await db.create_memory(device_memory)
                added_count += 1
                
                # Add to final memories
                final_memories.append(device_memory)
            
            # Mark as processed and synced
            processed_ids.add(memory_id)
            synced_ids.add(memory_id)
        
        # Add server memories not on device to final list
        for memory_id, server_memory in server_memory_map.items():
            if memory_id not in processed_ids:
                final_memories.append(server_memory)
        
        # Update sync state
        new_sync = SyncState(
            user_id=user_id,
            device_id=device_id,
            last_sync=now,
            synced_memory_ids=list(synced_ids)
        )
        sync_ref.set(new_sync.model_dump())
        
        # Create response
        response = SyncResponse(
            sync_id=sync_id,
            device_id=device_id,
            timestamp=now,
            added_count=added_count,
            updated_count=updated_count,
            deleted_count=deleted_count,
            conflicts=conflicts,
            server_memories=[m for m in server_memories if m.memory_id not in device_memory_map],
            last_sync=now
        )
        
        logger.info(f"Completed sync {sync_id} for user {user_id}, device {device_id}: "
                   f"added={added_count}, updated={updated_count}, deleted={deleted_count}, "
                   f"conflicts={len(conflicts)}")
        
        return response, final_memories
    
    except Exception as e:
        logger.error(f"Error during memory synchronization for user {user_id}: {str(e)}")
        raise DatabaseError(f"Failed to synchronize memories: {str(e)}")


@router.post("/sync", response_model=SyncResponse)
async def sync_endpoint(
    sync_request: SyncRequest,
    current_user: User = Depends(get_current_active_user)
) -> SyncResponse:
    """
    Synchronize memories between device and server.
    
    Args:
        sync_request: Synchronization request
        current_user: Current authenticated user
        
    Returns:
        SyncResponse: Synchronization results
        
    Raises:
        HTTPException: If synchronization fails
    """
    try:
        # Validate memory ownership
        for memory in sync_request.memories:
            if memory.user_id != current_user.id:
                logger.warning(f"Sync attempt with memory owned by {memory.user_id} from user {current_user.id}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Cannot sync memories owned by other users"
                )
        
        # Perform synchronization
        response, _ = await synchronize_memories(
            current_user.id,
            sync_request.device_id,
            sync_request.memories,
            sync_request.deleted_ids,
            sync_request.last_sync
        )
        
        return response
    
    except DatabaseError as e:
        logger.error(f"Database error during sync: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during sync: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sync failed: {str(e)}"
        )


@router.get("/sync/stats", response_model=Dict[str, SyncStats])
async def get_sync_stats(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, SyncStats]:
    """
    Get synchronization statistics for user devices.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict[str, SyncStats]: Stats for each device
        
    Raises:
        HTTPException: If stats retrieval fails
    """
    try:
        # Get Firestore client
        fs = get_firestore_client()
        
        # Query sync states for user
        sync_query = (fs.collection("sync_states")
                     .where("user_id", "==", current_user.id))
        
        # Collect stats for each device
        stats = {}
        for doc in sync_query.stream():
            sync_state = SyncState(**doc.to_dict())
            device_id = sync_state.device_id
            
            # Count memories for this device
            memory_query = (fs.collection("memories")
                           .where("user_id", "==", current_user.id))
            
            memory_count = len(list(memory_query.stream()))
            
            # Get sync history for frequency calculation
            history_query = (fs.collection("sync_history")
                            .where("user_id", "==", current_user.id)
                            .where("device_id", "==", device_id)
                            .order_by("timestamp", direction="DESCENDING")
                            .limit(10))
            
            sync_history = list(history_query.stream())
            total_syncs = len(sync_history)
            
            # Calculate average time between syncs
            sync_frequency = None
            if total_syncs >= 2:
                timestamps = [h.get("timestamp") for h in sync_history]
                intervals = [(timestamps[i] - timestamps[i+1]).total_seconds() / 3600
                            for i in range(len(timestamps)-1)]
                sync_frequency = sum(intervals) / len(intervals)
            
            # Create stats object
            stats[device_id] = SyncStats(
                device_id=device_id,
                last_sync=sync_state.last_sync,
                total_syncs=total_syncs,
                total_memories=len(sync_state.synced_memory_ids),
                memory_count=memory_count,
                sync_frequency=sync_frequency
            )
        
        return stats
    
    except Exception as e:
        logger.error(f"Error retrieving sync stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sync stats: {str(e)}"
        )


@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_memories(
    cleanup_request: CleanupRequest,
    current_user: User = Depends(get_current_active_user)
) -> CleanupResponse:
    """
    Clean up old memories and remove duplicates.
    
    Args:
        cleanup_request: Cleanup parameters
        current_user: Current authenticated user
        
    Returns:
        CleanupResponse: Cleanup results
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        # Get database and Firestore clients
        db = get_db_manager()
        fs = get_firestore_client()
        
        # Track results
        deleted_count = 0
        duplicate_count = 0
        affected_devices = set()
        
        # Run in dry-run mode if requested
        if cleanup_request.dry_run:
            logger.info(f"Performing DRY RUN cleanup for user {current_user.id}")
        
        # Clean up old memories
        cutoff_date = datetime.utcnow() - timedelta(days=cleanup_request.days_threshold)
        
        # Find old memories
        old_memories_query = (fs.collection("memories")
                             .where("user_id", "==", current_user.id)
                             .where("updated_at", "<", cutoff_date))
        
        old_memories = list(old_memories_query.stream())
        
        if not cleanup_request.dry_run:
            # Delete old memories
            for memory_doc in old_memories:
                memory_id = memory_doc.id
                await db.delete_memory(current_user.id, memory_id)
                deleted_count += 1
                
                # Update sync states to remove references
                sync_states_query = (fs.collection("sync_states")
                                    .where("user_id", "==", current_user.id))
                
                for sync_doc in sync_states_query.stream():
                    sync_data = sync_doc.to_dict()
                    device_id = sync_data.get("device_id")
                    synced_ids = set(sync_data.get("synced_memory_ids", []))
                    
                    if memory_id in synced_ids:
                        synced_ids.remove(memory_id)
                        sync_doc.reference.update({"synced_memory_ids": list(synced_ids)})
                        affected_devices.add(device_id)
        else:
            # Just count in dry-run mode
            deleted_count = len(old_memories)
            
            # Find affected devices
            for memory_doc in old_memories:
                memory_id = memory_doc.id
                sync_states_query = (fs.collection("sync_states")
                                   .where("user_id", "==", current_user.id))
                
                for sync_doc in sync_states_query.stream():
                    sync_data = sync_doc.to_dict()
                    device_id = sync_data.get("device_id")
                    synced_ids = set(sync_data.get("synced_memory_ids", []))
                    
                    if memory_id in synced_ids:
                        affected_devices.add(device_id)
        
        # Remove duplicates - specific device or all
        if cleanup_request.device_id:
            # Cleanup for specific device
            device_id = cleanup_request.device_id
            if not cleanup_request.dry_run:
                duplicate_count = await fs.cleanup_duplicate_memories(current_user.id)
                affected_devices.add(device_id)
            else:
                # Count duplicates in dry-run mode
                memories = await db.get_user_memories(current_user.id)
                content_map = {}
                for memory in memories:
                    content = memory.content
                    if content in content_map:
                        duplicate_count += 1
                    else:
                        content_map[content] = memory.memory_id
                
                affected_devices.add(device_id)
        else:
            # Cleanup for all devices
            if not cleanup_request.dry_run:
                duplicate_count = await fs.cleanup_duplicate_memories(current_user.id)
                
                # Get all affected devices
                sync_states_query = (fs.collection("sync_states")
                                    .where("user_id", "==", current_user.id))
                
                for sync_doc in sync_states_query.stream():
                    sync_data = sync_doc.to_dict()
                    device_id = sync_data.get("device_id")
                    affected_devices.add(device_id)
            else:
                # Count duplicates in dry-run mode
                memories = await db.get_user_memories(current_user.id)
                content_map = {}
                for memory in memories:
                    content = memory.content
                    if content in content_map:
                        duplicate_count += 1
                    else:
                        content_map[content] = memory.memory_id
                
                # Get all affected devices
                sync_states_query = (fs.collection("sync_states")
                                    .where("user_id", "==", current_user.id))
                
                for sync_doc in sync_states_query.stream():
                    sync_data = sync_doc.to_dict()
                    device_id = sync_data.get("device_id")
                    affected_devices.add(device_id)
        
        logger.info(f"Cleanup for user {current_user.id}: "
                   f"deleted={deleted_count}, duplicates={duplicate_count}, "
                   f"affected_devices={len(affected_devices)}, dry_run={cleanup_request.dry_run}")
        
        return CleanupResponse(
            deleted_count=deleted_count,
            duplicate_count=duplicate_count,
            affected_devices=list(affected_devices),
            dry_run=cleanup_request.dry_run
        )
    
    except Exception as e:
        logger.error(f"Error during memory cleanup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup memories: {str(e)}"
        )


@router.post("/reset-sync/{device_id}")
async def reset_sync_state(
    device_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Reset sync state for a device.
    
    Useful when a device needs to be re-initialized or
    sync problems need to be resolved.
    
    Args:
        device_id: Device ID to reset
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: Reset confirmation
        
    Raises:
        HTTPException: If reset fails
    """
    try:
        # Get Firestore client
        fs = get_firestore_client()
        
        # Find sync state for device
        sync_ref = (fs.collection("sync_states")
                   .document(f"{current_user.id}_{device_id}"))
        
        sync_doc = sync_ref.get()
        
        if not sync_doc.exists:
            logger.warning(f"Sync state not found for user {current_user.id}, device {device_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Sync state not found for this device"
            )
        
        # Create a reset sync state
        new_sync = SyncState(
            user_id=current_user.id,
            device_id=device_id,
            last_sync=None,
            synced_memory_ids=[]
        )
        
        # Update sync state
        sync_ref.set(new_sync.model_dump())
        
        logger.info(f"Reset sync state for user {current_user.id}, device {device_id}")
        
        return {
            "success": True,
            "message": f"Sync state reset for device {device_id}",
            "device_id": device_id,
            "timestamp": datetime.utcnow()
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error resetting sync state: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset sync state: {str(e)}"
        )