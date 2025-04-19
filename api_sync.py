"""
API Sync Router for EVA backend.

Provides endpoints for clients to synchronize data (primarily memories)
with the server, supporting offline-first approaches.
"""

import asyncio # Import asyncio if needed for other async operations
import logging
from datetime import datetime, timezone # Import timezone
from typing import Annotated, Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status

# --- Local Imports ---
from auth import get_current_active_user # Dependency for authentication
from database import get_db_manager, DatabaseManager # Use the main DB manager
from memory_manager import get_memory_manager, MemoryManager # Use memory manager if needed (e.g., for complex logic)
from models import User, Memory, SyncState # Import relevant models
from exceptions import DatabaseError, NotFoundException, AuthorizationError # Import needed exceptions

# --- Router Setup ---
router = APIRouter(
    prefix="/sync", # Added prefix for clarity
    tags=["Synchronization"], # Updated tag
    responses={
        401: {"description": "Unauthorized"},
        404: {"description": "Not found"},
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"}
    }
)

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
    last_sync_time: Optional[datetime] = Query(None, description="Timestamp of the last successful sync (ISO 8601 format, UTC)"),
    limit: int = Query(SYNC_BATCH_SIZE, ge=1, le=500, description="Maximum number of memories to retrieve"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None,
    db: DatabaseManager = Depends(get_db_manager)
) -> List[Memory]:
    """
    Retrieves memories for the current user that have been created or modified
    since the provided `last_sync_time`.

    If `last_sync_time` is omitted, retrieves all memories (up to the limit).
    Timestamps should be in UTC.
    """
    user_id = current_user.id
    logger.info(f"Sync request: Get memories for user {user_id} since {last_sync_time or 'beginning'}.")

    # Ensure last_sync_time is UTC if provided
    if last_sync_time and last_sync_time.tzinfo is None:
         logger.warning("last_sync_time provided without timezone, assuming UTC.")
         last_sync_time = last_sync_time.replace(tzinfo=timezone.utc)
    elif last_sync_time:
         last_sync_time = last_sync_time.astimezone(timezone.utc) # Convert to UTC

    try:
        # Call the new database method
        memories = await db.get_memories_since(
            user_id=user_id,
            last_sync_time=last_sync_time,
            limit=limit
        )

        logger.info(f"Sync: Returning {len(memories)} memories for user {user_id}.")
        return memories

    except DatabaseError as e: # Catch specific DB error
         logger.error(f"Database error retrieving memories for sync (User: {user_id}): {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error during sync.")
    except Exception as e:
         logger.error(f"Error retrieving memories for sync (User: {user_id}): {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve memories for sync.")

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
            # Although the client sends user_id, rely on the authenticated user
            client_memory.user_id = user_id # Force user_id to match authenticated user

            # --- Get existing server memory (if any) ---
            # Using DB directly for conflict check
            server_memory_data = await db.get_memory(memory_id)
            server_memory = Memory(**server_memory_data.model_dump()) if server_memory_data else None

            if server_memory:
                # --- Update Existing Memory ---
                # Ownership check (redundant if get_memory already checks, but safer)
                if server_memory.user_id != user_id:
                     logger.warning(f"Sync push rejected: Server memory {memory_id} ownership mismatch.")
                     failed_ids.append({"id": memory_id, "reason": "Ownership mismatch"})
                     continue

                # Conflict Resolution: Server timestamp wins if different
                # Ensure comparison is timezone-aware
                server_ts = server_memory.updated_at.astimezone(timezone.utc) if server_memory.updated_at.tzinfo else server_memory.updated_at.replace(tzinfo=timezone.utc)
                client_ts = client_memory.updated_at.astimezone(timezone.utc) if client_memory.updated_at.tzinfo else client_memory.updated_at.replace(tzinfo=timezone.utc)

                if server_ts > client_ts:
                    logger.warning(f"Sync conflict: Server memory {memory_id} is newer ({server_ts}) than client ({client_ts}). Ignoring client update.")
                    conflict_ids.append(memory_id)
                    continue # Skip update, server version is newer

                # Prepare updates dictionary from client memory
                # Exclude ID, ensure user_id is correct
                update_data = client_memory.model_dump(exclude={"memory_id"}, exclude_none=True)
                update_data["user_id"] = user_id # Ensure correct user ID
                # Mark as synced since the server is now processing this state
                update_data["is_synced"] = True
                # Use the client's updated_at timestamp to preserve modification time
                update_data["updated_at"] = client_ts # Use timezone-aware client timestamp

                # Use MemoryManager's update for potential validation/logic
                # Pass the dictionary of updates
                success = await mem_manager.update_memory(
                    memory_id=memory_id,
                    user_id=user_id,
                    updates=update_data
                )

                if success:
                    success_count += 1
                else:
                    logger.error(f"Sync push: Failed to update memory {memory_id} via MemoryManager.")
                    failed_ids.append({"id": memory_id, "reason": "Update failed"})

            else:
                # --- Create New Memory ---
                # Client is pushing a memory the server doesn't have
                logger.info(f"Sync push: Creating new memory {memory_id} from client.")
                # Ensure user_id is set correctly
                client_memory.user_id = user_id
                # Mark as synced since server is receiving it
                client_memory.is_synced = True
                # Ensure timestamp is timezone-aware UTC
                client_memory.updated_at = client_memory.updated_at.astimezone(timezone.utc) if client_memory.updated_at.tzinfo else client_memory.updated_at.replace(tzinfo=timezone.utc)
                client_memory.created_at = client_memory.created_at.astimezone(timezone.utc) if client_memory.created_at.tzinfo else client_memory.created_at.replace(tzinfo=timezone.utc)


                # Use MemoryManager or DB to create
                # Using DB directly here for simplicity, but manager might be better
                success = await db.create_memory(client_memory) # Pass the full object

                if success:
                    success_count += 1
                    # Update stats if using MemoryManager
                    # mem_manager._update_stats(MemoryType(client_memory.source), 1)
                else:
                    logger.error(f"Sync push: Failed to create new memory {memory_id} in DB.")
                    failed_ids.append({"id": memory_id, "reason": "Database creation failed"})

        except (AuthorizationError, NotFoundException) as e: # Catch errors from mem_manager.update_memory
             logger.warning(f"Sync push: Auth/Not Found error processing memory {memory_id}: {e}")
             failed_ids.append({"id": memory_id, "reason": str(e)})
        except Exception as e:
            logger.error(f"Sync push: Error processing memory {memory_id}: {e}", exc_info=True)
            failed_ids.append({"id": memory_id, "reason": f"Internal error: {type(e).__name__}"})

    logger.info(f"Sync push completed for user {user_id}: "
                f"{success_count} succeeded, {len(failed_ids)} failed, {len(conflict_ids)} conflicts.")

    return {
        "processed_count": len(memories_to_update),
        "success_count": success_count,
        "failed_memories": failed_ids,
        "conflicts_ignored": conflict_ids, # IDs where server version was newer
        "sync_timestamp": datetime.now(timezone.utc) # Provide server time for next sync
    }

# TODO: Add endpoints for other data types if needed (e.g., conversations, user settings)
# TODO: Consider adding a /sync/state endpoint to manage SyncState model if used.