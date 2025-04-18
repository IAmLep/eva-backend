"""
API Sync Router for EVA backend.

Provides endpoints for clients to synchronize data (primarily memories)
with the server, supporting offline-first approaches.
"""

import logging
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status

# --- Local Imports ---
from auth import get_current_active_user # Dependency for authentication
from database import get_db_manager, DatabaseManager # Use the main DB manager
from memory_manager import get_memory_manager, MemoryManager # Use memory manager if needed (e.g., for complex logic)
from models import User, Memory, SyncState # Import relevant models
from exceptions import DatabaseError, NotFoundException

# --- Router Setup ---
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# --- Sync Logic ---

# Constants for sync
SYNC_BATCH_SIZE = 100 # How many records to process per sync request

@router.get(
    "/memories",
    response_model=List[Memory], # Return a list of Memory objects
    summary="Get memories modified since last sync"
)
async def get_memories_for_sync(
    last_sync_time: Optional[datetime] = Query(None, description="Timestamp of the last successful sync (ISO 8601 format)"),
    limit: int = Query(SYNC_BATCH_SIZE, ge=1, le=500, description="Maximum number of memories to retrieve"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    db: DatabaseManager = Depends(get_db_manager)
) -> List[Memory]:
    """
    Retrieves memories for the current user that have been created or modified
    since the provided `last_sync_time`.

    If `last_sync_time` is omitted, retrieves all memories (up to the limit).
    """
    user_id = current_user.id
    logger.info(f"Sync request: Get memories for user {user_id} since {last_sync_time or 'beginning'}.")

    try:
        # --- Firestore Query ---
        # Query needs to filter by user_id and updated_at > last_sync_time
        # Order by updated_at to ensure consistency
        if db.db and auth.FieldFilter: # Check if Firestore is active and FieldFilter available
            query = db.db.collection("memories").where(filter=auth.FieldFilter("user_id", "==", user_id))
            if last_sync_time:
                # Ensure last_sync_time is timezone-aware (UTC) if needed
                query = query.where(filter=auth.FieldFilter("updated_at", ">", last_sync_time))
            query = query.order_by("updated_at", direction=auth.firebase_firestore.Query.ASCENDING).limit(limit)
            results = await asyncio.to_thread(query.stream)
            memories = [Memory(**doc.to_dict()) for doc in results]

        elif not db.db: # In-memory fallback
             user_mems = [Memory(**m) for m in db.in_memory_db["memories"].values() if m.get("user_id") == user_id]
             if last_sync_time:
                  user_mems = [m for m in user_mems if m.updated_at > last_sync_time]
             user_mems.sort(key=lambda m: m.updated_at) # Sort ascending
             memories = user_mems[:limit]
        else:
            # Firestore available but FieldFilter missing (older library version?)
            logger.error("Firestore FieldFilter not available, cannot perform time-based sync query.")
            raise HTTPException(status_code=501, detail="Server configuration does not support time-based sync query.")


        logger.info(f"Sync: Returning {len(memories)} memories for user {user_id}.")
        return memories

    except Exception as e:
        logger.error(f"Error retrieving memories for sync (User: {user_id}): {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories for sync: {e}"
        )

@router.post(
    "/memories",
    status_code=status.HTTP_200_OK,
    summary="Push client-side memory changes to server"
)
async def push_memory_changes(
    memories_to_update: List[Memory], # Client sends full Memory objects
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    db: DatabaseManager = Depends(get_db_manager),
    mem_manager: MemoryManager = Depends(get_memory_manager) # Inject MemoryManager too
) -> Dict[str, Any]:
    """
    Receives a list of memories created or modified on the client and
    updates the server-side storage.

    - Checks ownership.
    - Compares `updated_at` timestamps to handle conflicts (server wins).
    - Creates new memories or updates existing ones.
    - Returns a summary of successes and failures.
    """
    user_id = current_user.id
    success_count = 0
    failed_ids = []
    conflict_ids = []

    logger.info(f"Sync request: Push {len(memories_to_update)} memory changes for user {user_id}.")

    if len(memories_to_update) > SYNC_BATCH_SIZE * 2: # Limit batch size
        raise HTTPException(status_code=413, detail=f"Payload too large. Max {SYNC_BATCH_SIZE * 2} memories per request.")

    for client_memory in memories_to_update:
        memory_id = client_memory.memory_id
        try:
            # --- Security Check: Ensure memory belongs to the current user ---
            if client_memory.user_id != user_id:
                logger.warning(f"Sync push rejected: Memory {memory_id} user_id mismatch "
                               f"(Client: {client_memory.user_id}, Auth: {user_id})")
                failed_ids.append({"id": memory_id, "reason": "User ID mismatch"})
                continue

            # --- Get existing server memory (if any) ---
            # Use MemoryManager's get_memory for potential caching/logic, or DB directly
            # Using DB directly here for simplicity
            server_memory = await db.get_memory(memory_id)

            if server_memory:
                # --- Update Existing Memory ---
                # Conflict Resolution: Server timestamp wins if different
                if server_memory.updated_at > client_memory.updated_at:
                    logger.warning(f"Sync conflict: Server memory {memory_id} is newer. Ignoring client update.")
                    conflict_ids.append(memory_id)
                    continue # Skip update, server version is newer

                # Prepare updates dictionary from client memory
                # Ensure updated_at reflects the client's latest change time
                update_data = client_memory.model_dump(exclude={"memory_id"}, exclude_none=True)
                # Mark as synced since the server is now processing this state
                update_data["is_synced"] = True
                # Explicitly set updated_at from client? Or use server time now?
                # Using client's updated_at allows preserving modification time across syncs
                # update_data["updated_at"] = client_memory.updated_at # Keep client time

                # Use MemoryManager's update for potential validation/logic
                # Or DB directly
                # success = await mem_manager.update_memory(memory_id, user_id, update_data)
                # Using DB directly requires ensuring ownership again if needed
                success = await db.update_memory(memory_id, update_data)

                if success:
                    success_count += 1
                else:
                    logger.error(f"Sync push: Failed to update memory {memory_id} in DB.")
                    failed_ids.append({"id": memory_id, "reason": "Database update failed"})

            else:
                # --- Create New Memory ---
                # Client is pushing a memory the server doesn't have
                logger.info(f"Sync push: Creating new memory {memory_id} from client.")
                # Mark as synced since server is receiving it
                client_memory.is_synced = True
                # Use MemoryManager or DB to create
                # success = await mem_manager.create_memory(...) # Needs splitting data
                success = await db.create_memory(client_memory) # Pass the full object

                if success:
                    success_count += 1
                else:
                    logger.error(f"Sync push: Failed to create new memory {memory_id} in DB.")
                    failed_ids.append({"id": memory_id, "reason": "Database creation failed"})

        except Exception as e:
            logger.error(f"Sync push: Error processing memory {memory_id}: {e}", exc_info=True)
            failed_ids.append({"id": memory_id, "reason": str(e)})

    logger.info(f"Sync push completed for user {user_id}: "
                f"{success_count} succeeded, {len(failed_ids)} failed, {len(conflict_ids)} conflicts.")

    return {
        "processed_count": len(memories_to_update),
        "success_count": success_count,
        "failed_memories": failed_ids,
        "conflicts_ignored": conflict_ids, # IDs where server version was newer
        "sync_timestamp": datetime.now(auth.timezone.utc) # Provide server time for next sync
    }

# TODO: Add endpoints for other data types if needed (e.g., conversations, user settings)
# TODO: Consider adding a /sync/state endpoint to manage SyncState model if used.