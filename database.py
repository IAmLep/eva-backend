"""
Database Manager for EVA backend using Firestore.

Provides an abstraction layer for interacting with the Firestore database,
handling operations for users, memories (core, event, conversational),
conversations, sync states, API keys, and secret categories.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone # Import timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# --- Google Cloud Firestore ---
try:
    import firebase_admin
    from firebase_admin import credentials, firestore as firebase_firestore
    try:
        from google.cloud.firestore_v1.base_query import FieldFilter
    except ImportError:
        from google.cloud.firestore_v1.query import FieldFilter
    # Alias Query to avoid potential conflicts
    from google.cloud.firestore_v1.query import Query as FirestoreQuery
    from google.api_core import exceptions as google_exceptions

    FIRESTORE_AVAILABLE = True
    # Flag to check if FieldFilter is usable (for older library versions)
    FIRESTORE_FIELD_FILTER_AVAILABLE = hasattr(FieldFilter, 'from_string') or hasattr(FieldFilter, '__init__')

except ImportError:
    FIRESTORE_AVAILABLE = False
    FIRESTORE_FIELD_FILTER_AVAILABLE = False
    # Define dummy types for fallback if Firestore is not installed
    firebase_admin = None
    firebase_firestore = None
    FieldFilter = None
    FirestoreQuery = None
    google_exceptions = None
    logger.warning("Firestore libraries not found. Using in-memory database fallback.")


# --- Local Imports ---
from config import settings
from models import Memory, User, UserInDB, Conversation, SyncState, ApiKey, UserRateLimit, MemorySource, MemoryCategory # Added UserInDB
from exceptions import DatabaseError, ConfigurationError, NotFoundException # Import NotFoundException

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        """Initialize database connection (Firestore or in-memory)."""
        self.settings = settings # <-- CORRECTED LINE 57
        self.db = None
        self.in_memory_db = {
            "users": {}, "memories": {}, "conversations": {},
            "sync_states": {}, "api_keys": {}, "categories": {}, "secrets": {}
        } # For fallback

        if FIRESTORE_AVAILABLE:
            try:
                # Use self.settings now
                if self.settings.USE_GCP_DEFAULT_CREDENTIALS:
                    # Use Application Default Credentials (ADC) - preferred for Cloud Run/GCP
                    cred = None # firebase_admin uses ADC if cred is None
                    logger.info("Initializing Firestore using Application Default Credentials.")
                elif self.settings.FIREBASE_CREDENTIALS_PATH and os.path.exists(self.settings.FIREBASE_CREDENTIALS_PATH):
                    # Use service account key file
                    cred = credentials.Certificate(self.settings.FIREBASE_CREDENTIALS_PATH)
                    logger.info(f"Initializing Firestore using credentials file: {self.settings.FIREBASE_CREDENTIALS_PATH}")
                else:
                    raise ConfigurationError("Firestore credentials not configured. Set FIREBASE_CREDENTIALS_PATH or enable USE_GCP_DEFAULT_CREDENTIALS.")

                # Initialize Firebase Admin SDK (only once)
                if not firebase_admin._apps:
                    # Use self.settings now
                    firebase_admin.initialize_app(cred, {
                        'projectId': self.settings.FIREBASE_PROJECT_ID,
                    })

                self.db = firebase_firestore.client()
                # Use self.settings now
                logger.info(f"Firestore client initialized for project: {self.settings.FIREBASE_PROJECT_ID}")

            except FileNotFoundError:
                 # Use self.settings now
                 logger.error(f"Firebase credentials file not found at {self.settings.FIREBASE_CREDENTIALS_PATH}. Using in-memory DB.")
                 self.db = None
            except ConfigurationError as e:
                 logger.error(f"{e}. Using in-memory DB.")
                 self.db = None
            except Exception as e:
                logger.exception(f"Failed to initialize Firestore: {e}. Using in-memory DB.", exc_info=e)
                self.db = None
        else:
            logger.warning("Firestore libraries not installed. Using in-memory database.")
            self.db = None # Ensure db is None if import failed

    def _get_collection(self, name: str):
        """Helper to get a Firestore collection reference."""
        if not self.db:
            raise DatabaseError("Database connection is not available.")
        return self.db.collection(name)

    # --- User Operations ---
    async def create_user(self, user: UserInDB) -> str: # Expect UserInDB now
        user_id = user.id # Use the ID generated by the model's default_factory
        # Store salt if provided (important if deriving keys later)
        user_data = user.model_dump(exclude={"id"}, exclude_none=True)
        try:
            if self.db:
                doc_ref = self.db.collection("users").document(user_id)
                await asyncio.to_thread(doc_ref.set, user_data)
            else:
                self.in_memory_db["users"][user_id] = user_data
            logger.info(f"Created user {user_id}: {user.username}")
            return user_id
        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create user: {e}")

    async def get_user(self, user_id: str) -> Optional[UserInDB]: # Return UserInDB
        try:
            if self.db:
                doc_ref = self.db.collection("users").document(user_id)
                doc = await asyncio.to_thread(doc_ref.get)
                if doc.exists:
                    user_data = doc.to_dict()
                    user_data["id"] = doc.id
                    # Ensure all UserInDB fields are present or have defaults
                    return UserInDB(**user_data)
                return None
            else: # In-memory
                user_data = self.in_memory_db["users"].get(user_id)
                if user_data:
                    user_data["id"] = user_id
                    return UserInDB(**user_data)
                return None
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {e}", exc_info=True)
            return None # Return None on error

    async def get_user_by_field(self, field_name: str, value: Any) -> Optional[UserInDB]: # Return UserInDB
         try:
             if self.db:
                  if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                       logger.error("Firestore FieldFilter not available, cannot query by field.")
                       # Fallback: Fetch all and filter? Very inefficient.
                       return None # Or raise NotImplementedError

                  query = self.db.collection("users").where(filter=FieldFilter(field_name, "==", value)).limit(1)
                  results = await asyncio.to_thread(query.stream)
                  for doc in results: # Should be at most one due to limit(1)
                       user_data = doc.to_dict()
                       user_data["id"] = doc.id
                       return UserInDB(**user_data) # Cast to UserInDB
                  return None
             else: # In-memory
                  for user_id, user_data in self.in_memory_db["users"].items():
                       if user_data.get(field_name) == value:
                            user_data["id"] = user_id
                            return UserInDB(**user_data)
                  return None
         except Exception as e:
              logger.error(f"Error retrieving user by {field_name}='{value}': {e}", exc_info=True)
              return None

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]: # Return UserInDB
        return await self.get_user_by_field("username", username)

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]: # Return UserInDB
        return await self.get_user_by_field("email", email)

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Updates user data in the database."""
        if "id" in updates: del updates["id"] # Cannot update ID
        updates["updated_at"] = datetime.now(timezone.utc) # Ensure timestamp is updated

        try:
            if self.db:
                doc_ref = self.db.collection("users").document(user_id)
                await asyncio.to_thread(doc_ref.update, updates)
            else: # In-memory
                if user_id in self.in_memory_db["users"]:
                    self.in_memory_db["users"][user_id].update(updates)
                else:
                    return False # User not found
            return True
        except google_exceptions.NotFound:
             logger.warning(f"Attempted to update non-existent user: {user_id}")
             return False
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update user: {e}")

    async def delete_user(self, user_id: str) -> bool:
        """Deletes a user from the database."""
        try:
            if self.db:
                doc_ref = self.db.collection("users").document(user_id)
                await asyncio.to_thread(doc_ref.delete)
                # TODO: Consider deleting associated memories, keys, etc. (cascade delete logic)
            else: # In-memory
                if user_id in self.in_memory_db["users"]:
                    del self.in_memory_db["users"][user_id]
                    # TODO: Cascade delete for in-memory
                else:
                    return False # User not found
            logger.info(f"Deleted user {user_id}")
            return True
        except google_exceptions.NotFound:
             logger.warning(f"Attempted to delete non-existent user: {user_id}")
             return False # Or True, depending on desired idempotency
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete user: {e}")

    # --- Memory Operations ---
    async def create_memory(self, memory: Memory) -> bool:
        """Creates a new memory document."""
        memory_id = memory.memory_id
        memory_data = memory.model_dump(exclude={"memory_id"}, exclude_none=True)
        try:
            if self.db:
                doc_ref = self.db.collection("memories").document(memory_id)
                await asyncio.to_thread(doc_ref.set, memory_data)
            else:
                self.in_memory_db["memories"][memory_id] = memory_data
            return True
        except Exception as e:
            logger.error(f"Error creating memory {memory_id}: {e}", exc_info=True)
            # Don't raise here, allow MemoryManager to handle
            return False

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Gets a specific memory document."""
        try:
            if self.db:
                doc_ref = self.db.collection("memories").document(memory_id)
                doc = await asyncio.to_thread(doc_ref.get)
                if doc.exists:
                    mem_data = doc.to_dict()
                    mem_data["memory_id"] = doc.id
                    return Memory(**mem_data)
                return None
            else: # In-memory
                mem_data = self.in_memory_db["memories"].get(memory_id)
                if mem_data:
                    mem_data["memory_id"] = memory_id
                    return Memory(**mem_data)
                return None
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}", exc_info=True)
            return None

    async def get_user_memories(self, user_id: str, limit: int = 50, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]:
        """Gets memories for a user, with optional filters."""
        memories = []
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                     logger.error("Firestore FieldFilter not available, cannot query memories by field.")
                     return [] # Or raise

                query = self.db.collection("memories").where(filter=FieldFilter("user_id", "==", user_id))
                if memory_type:
                    query = query.where(filter=FieldFilter("source", "==", memory_type))
                if tags:
                    # Firestore 'array-contains-any' is good for OR, 'array-contains' for AND (single tag)
                    # For multiple AND tags, you might need multiple queries or client-side filtering
                    # Using 'array-contains-any' for broader matching
                    query = query.where(filter=FieldFilter("tags", "array_contains_any", tags))

                # Add ordering and pagination
                query = query.order_by("updated_at", direction=FirestoreQuery.DESCENDING) # Order by most recent
                if offset > 0:
                     # Firestore pagination requires using start_after with the last doc of previous page
                     # Simple offset isn't directly supported. Fetching limit+offset and slicing is inefficient.
                     # For now, ignoring offset for Firestore due to complexity.
                     logger.warning("Firestore query ignoring 'offset' parameter.")
                     query = query.limit(limit)
                else:
                     query = query.limit(limit)

                results = await asyncio.to_thread(query.stream)
                for doc in results:
                    mem_data = doc.to_dict()
                    mem_data["memory_id"] = doc.id
                    memories.append(Memory(**mem_data))

            else: # In-memory
                user_mems = [Memory(**m, memory_id=m_id) for m_id, m in self.in_memory_db["memories"].items() if m.get("user_id") == user_id]
                if memory_type:
                    user_mems = [m for m in user_mems if m.source.value == memory_type]
                if tags:
                    user_mems = [m for m in user_mems if any(tag in m.tags for tag in tags)]
                user_mems.sort(key=lambda m: m.updated_at, reverse=True)
                memories = user_mems[offset:offset+limit]

            return memories
        except Exception as e:
            logger.error(f"Error retrieving user memories for {user_id}: {e}", exc_info=True)
            return [] # Return empty list on error

    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Updates a memory document."""
        if "memory_id" in updates: del updates["memory_id"]
        updates["updated_at"] = datetime.now(timezone.utc) # Ensure timestamp

        try:
            if self.db:
                doc_ref = self.db.collection("memories").document(memory_id)
                await asyncio.to_thread(doc_ref.update, updates)
            else: # In-memory
                if memory_id in self.in_memory_db["memories"]:
                    self.in_memory_db["memories"][memory_id].update(updates)
                else:
                    return False
            return True
        except google_exceptions.NotFound:
             logger.warning(f"Attempted to update non-existent memory: {memory_id}")
             return False
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}", exc_info=True)
            return False # Indicate failure

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Deletes a memory document. Optionally checks user_id."""
        # Note: Ownership check should happen in MemoryManager before calling this
        try:
            if self.db:
                doc_ref = self.db.collection("memories").document(memory_id)
                # Optional: Get doc first to verify user_id if needed at DB layer
                # doc = await asyncio.to_thread(doc_ref.get)
                # if not doc.exists or doc.to_dict().get("user_id") != user_id: return False
                await asyncio.to_thread(doc_ref.delete)
            else: # In-memory
                if memory_id in self.in_memory_db["memories"]:
                    # Optional: Verify user_id
                    # if self.in_memory_db["memories"][memory_id].get("user_id") != user_id: return False
                    del self.in_memory_db["memories"][memory_id]
                else:
                    return False
            return True
        except google_exceptions.NotFound:
             logger.warning(f"Attempted to delete non-existent memory: {memory_id}")
             return False # Or True for idempotency
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}", exc_info=True)
            return False

    async def get_events_by_timerange(self, user_id: str, start_time: datetime, end_time: datetime) -> List[Memory]:
        """Gets event memories within a specific time range."""
        memories = []
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                     logger.error("Firestore FieldFilter not available, cannot query events by time range.")
                     return []

                # Query based on the 'event_time' stored in metadata
                query = self.db.collection("memories") \
                    .where(filter=FieldFilter("user_id", "==", user_id)) \
                    .where(filter=FieldFilter("source", "==", MemorySource.EVENT.value)) \
                    .where(filter=FieldFilter("metadata.event_time", ">=", start_time.isoformat())) \
                    .where(filter=FieldFilter("metadata.event_time", "<=", end_time.isoformat())) \
                    .order_by("metadata.event_time", direction=FirestoreQuery.ASCENDING)

                results = await asyncio.to_thread(query.stream)
                for doc in results:
                    mem_data = doc.to_dict()
                    mem_data["memory_id"] = doc.id
                    memories.append(Memory(**mem_data))
            else: # In-memory
                user_mems = [Memory(**m, memory_id=m_id) for m_id, m in self.in_memory_db["memories"].items()
                             if m.get("user_id") == user_id and m.get("source") == MemorySource.EVENT.value]
                for mem in user_mems:
                    event_time_str = mem.metadata.get("event_time")
                    try:
                        # Ensure timezone comparison is correct
                        event_dt_naive = datetime.fromisoformat(event_time_str.replace('Z', ''))
                        event_dt = event_dt_naive.replace(tzinfo=timezone.utc) # Assume UTC if Z present
                        # Ensure start/end times are timezone-aware (preferably UTC)
                        if start_time.tzinfo is None: start_time = start_time.replace(tzinfo=timezone.utc)
                        if end_time.tzinfo is None: end_time = end_time.replace(tzinfo=timezone.utc)

                        if start_time <= event_dt <= end_time:
                            memories.append(mem)
                    except (TypeError, ValueError):
                        logger.warning(f"Skipping memory {mem.memory_id} due to invalid event_time format: {event_time_str}")
                        continue # Skip if event_time is invalid
                memories.sort(key=lambda m: m.metadata.get("event_time", "9999"))

            return memories
        except Exception as e:
            logger.error(f"Error retrieving events by time range for user {user_id}: {e}", exc_info=True)
            return []

    # --- ADDED: get_memories_since for Sync ---
    async def get_memories_since(self, user_id: str, last_sync_time: Optional[datetime], limit: int) -> List[Memory]:
        """Retrieves memories updated since a given time."""
        memories = []
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                    # Fallback: Fetch recent N and filter in memory (less efficient)
                    logger.warning("Firestore FieldFilter not available. Sync query will be less efficient.")
                    query = self.db.collection("memories").where(filter=FieldFilter("user_id", "==", user_id))
                    # Fetch more than limit and sort/filter locally
                    query = query.order_by("updated_at", direction=FirestoreQuery.DESCENDING).limit(limit * 5) # Example: Fetch 5x
                    results = await asyncio.to_thread(query.stream)
                    all_user_mems = [Memory(**doc.to_dict(), memory_id=doc.id) for doc in results]

                    # Ensure last_sync_time is timezone-aware (UTC) for comparison
                    if last_sync_time and last_sync_time.tzinfo is None:
                        last_sync_time = last_sync_time.replace(tzinfo=timezone.utc)

                    if last_sync_time:
                        memories = [m for m in all_user_mems if m.updated_at > last_sync_time]
                    else: # First sync, return all fetched
                        memories = all_user_mems

                    memories.sort(key=lambda m: m.updated_at) # Sort ascending for consistency
                    memories = memories[:limit] # Apply limit after filtering

                else: # FieldFilter is available
                    query = self.db.collection("memories").where(filter=FieldFilter("user_id", "==", user_id))

                    # Ensure last_sync_time is timezone-aware (UTC) for Firestore query
                    if last_sync_time and last_sync_time.tzinfo is None:
                        last_sync_time = last_sync_time.replace(tzinfo=timezone.utc)

                    if last_sync_time:
                        # Ensure timestamp comparison works correctly
                        query = query.where(filter=FieldFilter("updated_at", ">", last_sync_time))
                    # Order by updated_at ASCENDING for proper sync sequence
                    query = query.order_by("updated_at", direction=FirestoreQuery.ASCENDING).limit(limit)
                    results = await asyncio.to_thread(query.stream)
                    memories = [Memory(**doc.to_dict(), memory_id=doc.id) for doc in results]

            else: # In-memory fallback
                # Ensure memory objects are created for comparison
                user_mems = [Memory(**m, memory_id=m_id) for m_id, m in self.in_memory_db["memories"].items() if m.get("user_id") == user_id]

                # Ensure last_sync_time is timezone-aware (UTC) for comparison
                if last_sync_time and last_sync_time.tzinfo is None:
                    last_sync_time = last_sync_time.replace(tzinfo=timezone.utc)

                if last_sync_time:
                    user_mems = [m for m in user_mems if m.updated_at > last_sync_time]
                user_mems.sort(key=lambda m: m.updated_at) # Sort ascending
                memories = user_mems[:limit]

            return memories
        except Exception as e:
            logger.error(f"Error in get_memories_since for user {user_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve memories since {last_sync_time}: {e}")


    async def get_memories_by_query(self, user_id: str, query: str, limit: int = 10) -> List[Memory]:
        """Placeholder: Retrieves memories based on a simple text query."""
        logger.warning("get_memories_by_query using simple keyword matching.")
        query_terms = set(q.lower() for q in query.split() if len(q) > 2)
        all_memories = await self.get_user_memories(user_id, limit=200) # Fetch a larger pool
        matched = []
        for mem in all_memories:
            content_lower = mem.content.lower()
            if any(term in content_lower for term in query_terms):
                matched.append(mem)
                if len(matched) >= limit:
                    break
        # Optional: Sort matched memories by relevance or date?
        matched.sort(key=lambda m: m.updated_at, reverse=True)
        return matched[:limit] # Ensure limit is applied

    async def cleanup_old_memories(self, user_id: str, days_threshold: int) -> int:
        """Placeholder: Deletes memories older than the specified threshold."""
        logger.warning(f"Placeholder cleanup_old_memories called for user {user_id}, threshold {days_threshold} days. No deletion performed.")
        # Implementation would involve querying memories with updated_at < (now - threshold)
        # and deleting them in batches.
        return 0

    async def cleanup_duplicate_memories(self, user_id: str) -> int:
        """Placeholder: Deletes duplicate memories based on content hash or similarity."""
        logger.warning(f"Placeholder cleanup_duplicate_memories called for user {user_id}. No deletion performed.")
        # Implementation would involve fetching memories, calculating hashes (e.g., hashlib.sha256(m.content.encode()).hexdigest()),
        # identifying duplicates, keeping one, and deleting others.
        return 0

    # --- Conversation Operations (Stubs - Implement as needed) ---
    async def create_conversation(self, conversation: Conversation) -> bool:
        logger.warning("create_conversation not implemented.")
        return False
    async def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        logger.warning("get_conversation not implemented.")
        return None
    async def update_conversation(self, conversation_id: str, updates: Dict[str, Any]) -> bool:
        logger.warning("update_conversation not implemented.")
        return False

    # --- Sync Operations (Stubs - Implement as needed) ---
    async def get_sync_state(self, user_id: str, device_id: str) -> Optional[SyncState]:
        logger.warning("get_sync_state not implemented.")
        return None
    async def update_sync_state(self, sync_state: SyncState) -> bool:
        logger.warning("update_sync_state not implemented.")
        return False

    # --- API Key Operations ---
    async def get_api_key_by_hash(self, hashed_key: str) -> Optional[ApiKey]:
        # This might be less efficient if not indexed well on hashed_key
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                     logger.error("Firestore FieldFilter not available, cannot query API keys by hash.")
                     return None
                # Ensure hashed_key is indexed in Firestore for performance
                query = self.db.collection("api_keys").where(filter=FieldFilter("hashed_key", "==", hashed_key)).limit(1)
                results = await asyncio.to_thread(query.stream)
                for doc in results:
                    key_data = doc.to_dict()
                    key_data["key_id"] = doc.id
                    return ApiKey(**key_data)
                return None
            else: # In-memory
                for key_id, key_data in self.in_memory_db["api_keys"].items():
                    if key_data.get("hashed_key") == hashed_key:
                        key_data_copy = key_data.copy() # Avoid modifying original
                        key_data_copy["key_id"] = key_id
                        return ApiKey(**key_data_copy)
                return None
        except Exception as e:
            logger.error(f"Error retrieving API key by hash: {e}", exc_info=True)
            return None

    async def get_api_keys_by_prefix(self, prefix: str) -> List[ApiKey]:
        """Gets API keys matching a given prefix."""
        # This query requires an index on 'prefix' in Firestore.
        keys = []
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                    logger.error("Firestore FieldFilter not available, cannot query API keys by prefix.")
                    return []
                query = self.db.collection("api_keys").where(filter=FieldFilter("prefix", "==", prefix))
                results = await asyncio.to_thread(query.stream)
                for doc in results:
                    key_data = doc.to_dict()
                    key_data["key_id"] = doc.id
                    keys.append(ApiKey(**key_data))
            else: # In-memory
                for key_id, key_data in self.in_memory_db["api_keys"].items():
                    if key_data.get("prefix") == prefix:
                        key_data_copy = key_data.copy() # Avoid modifying original dict
                        key_data_copy["key_id"] = key_id
                        keys.append(ApiKey(**key_data_copy))
            logger.debug(f"Retrieved {len(keys)} API keys with prefix '{prefix}'")
            return keys
        except Exception as e:
            logger.error(f"Error retrieving API keys by prefix '{prefix}': {e}", exc_info=True)
            return [] # Return empty list on error


    async def update_api_key_usage(self, key_id: str) -> bool:
        """Updates the last_used timestamp for an API key."""
        try:
            update = {"last_used": datetime.now(timezone.utc)}
            if self.db:
                doc_ref = self.db.collection("api_keys").document(key_id)
                await asyncio.to_thread(doc_ref.update, update)
            else:
                if key_id in self.in_memory_db["api_keys"]:
                    self.in_memory_db["api_keys"][key_id].update(update)
                else:
                    logger.warning(f"Attempted to update usage for non-existent in-memory API key: {key_id}")
                    return False
            return True
        except google_exceptions.NotFound:
             logger.warning(f"Attempted to update usage for non-existent Firestore API key: {key_id}")
             return False
        except Exception as e:
            logger.error(f"Error updating API key usage for {key_id}: {e}", exc_info=True)
            return False

    # --- Secret Category & Secret Operations ---
    # (Assuming implementations using self.db or self.in_memory_db are correct,
    #  but adding basic ownership checks and error handling consistency)

    async def create_category(self, category_data: Dict[str, Any]) -> bool:
        cat_id = category_data.get("id")
        if not cat_id:
             logger.error("Category data missing 'id'.")
             return False
        try:
            if self.db: await asyncio.to_thread(self.db.collection("categories").document(cat_id).set, category_data)
            else: self.in_memory_db["categories"][cat_id] = category_data.copy() # Store a copy
            return True
        except Exception as e:
             logger.error(f"DB Error create_category {cat_id}: {e}", exc_info=True)
             return False

    async def get_category(self, user_id: str, category_id: str) -> Optional[Dict[str, Any]]:
        try:
            if self.db:
                doc_ref = self.db.collection("categories").document(category_id)
                doc = await asyncio.to_thread(doc_ref.get)
                if doc.exists:
                     data = doc.to_dict()
                     if data.get("user_id") == user_id:
                          data['id'] = doc.id # Ensure ID is included
                          return data
                return None # Not found or wrong user
            else: # In-memory
                data = self.in_memory_db["categories"].get(category_id)
                if data and data.get("user_id") == user_id:
                     data_copy = data.copy()
                     data_copy['id'] = category_id # Ensure ID is included
                     return data_copy
                return None
        except Exception as e:
             logger.error(f"DB Error get_category {category_id}: {e}", exc_info=True)
             return None

    async def get_user_categories(self, user_id: str) -> List[Dict[str, Any]]:
        cats = []
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                     logger.error("Firestore FieldFilter not available for get_user_categories.")
                     return []
                query = self.db.collection("categories").where(filter=FieldFilter("user_id", "==", user_id))
                results = await asyncio.to_thread(query.stream)
                for doc in results:
                     cat_data = doc.to_dict()
                     cat_data['id'] = doc.id # Ensure ID is included
                     cats.append(cat_data)
            else: # In-memory
                for cat_id, cat_data in self.in_memory_db["categories"].items():
                     if cat_data.get("user_id") == user_id:
                          cat_data_copy = cat_data.copy()
                          cat_data_copy['id'] = cat_id # Ensure ID is included
                          cats.append(cat_data_copy)
            return cats
        except Exception as e:
             logger.error(f"DB Error get_user_categories for user {user_id}: {e}", exc_info=True)
             return []

    async def delete_category(self, user_id: str, category_id: str) -> bool:
        # Perform ownership check before deleting
        category_to_delete = await self.get_category(user_id, category_id)
        if not category_to_delete:
            logger.warning(f"Attempt to delete non-existent or unauthorized category {category_id} by user {user_id}")
            return False # Not found or not owned by user

        try:
            if self.db:
                await asyncio.to_thread(self.db.collection("categories").document(category_id).delete)
                # TODO: Delete associated secrets? Requires a query.
            else: # In-memory
                self.in_memory_db["categories"].pop(category_id, None)
                # TODO: Delete associated secrets for in-memory.
            logger.info(f"Deleted category {category_id} for user {user_id}")
            return True
        except Exception as e:
             logger.error(f"DB Error delete_category {category_id}: {e}", exc_info=True)
             return False

    async def create_secret(self, secret_data: Dict[str, Any]) -> bool:
        secret_id = secret_data.get("id")
        if not secret_id:
             logger.error("Secret data missing 'id'.")
             return False
        try:
            if self.db: await asyncio.to_thread(self.db.collection("secrets").document(secret_id).set, secret_data)
            else: self.in_memory_db["secrets"][secret_id] = secret_data.copy() # Store a copy
            return True
        except Exception as e:
             logger.error(f"DB Error create_secret {secret_id}: {e}", exc_info=True)
             return False

    async def get_secret(self, user_id: str, secret_id: str) -> Optional[Dict[str, Any]]:
        # Perform ownership check
        try:
            if self.db:
                doc_ref = self.db.collection("secrets").document(secret_id)
                doc = await asyncio.to_thread(doc_ref.get)
                if doc.exists:
                     data = doc.to_dict()
                     if data.get("user_id") == user_id:
                          data['id'] = doc.id # Ensure ID included
                          return data
                return None # Not found or wrong user
            else: # In-memory
                data = self.in_memory_db["secrets"].get(secret_id)
                if data and data.get("user_id") == user_id:
                     data_copy = data.copy()
                     data_copy['id'] = secret_id # Ensure ID included
                     return data_copy
                return None
        except Exception as e:
             logger.error(f"DB Error get_secret {secret_id}: {e}", exc_info=True)
             return None

    async def get_user_secrets(self, user_id: str, category_id: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        secrets_list = []
        try:
            if self.db:
                if not FIRESTORE_FIELD_FILTER_AVAILABLE:
                     logger.error("Firestore FieldFilter not available for get_user_secrets.")
                     return []
                query = self.db.collection("secrets").where(filter=FieldFilter("user_id", "==", user_id))
                if category_id: query = query.where(filter=FieldFilter("category_id", "==", category_id))
                if tag: query = query.where(filter=FieldFilter("tags", "array_contains", tag)) # Requires index on tags
                results = await asyncio.to_thread(query.stream)
                for doc in results:
                     secret_data = doc.to_dict()
                     secret_data['id'] = doc.id # Ensure ID included
                     secrets_list.append(secret_data)
            else: # In-memory
                user_secrets = []
                for secret_id, secret_data in self.in_memory_db["secrets"].items():
                     if secret_data.get("user_id") == user_id:
                          matches_category = (not category_id) or (secret_data.get("category_id") == category_id)
                          matches_tag = (not tag) or (tag in secret_data.get("tags", []))
                          if matches_category and matches_tag:
                               secret_data_copy = secret_data.copy()
                               secret_data_copy['id'] = secret_id # Ensure ID included
                               user_secrets.append(secret_data_copy)
                secrets_list = user_secrets
            return secrets_list
        except Exception as e:
             logger.error(f"DB Error get_user_secrets for user {user_id}: {e}", exc_info=True)
             return []

    async def update_secret(self, user_id: str, secret_id: str, updates: Dict[str, Any]) -> bool:
        # Perform ownership check before updating
        secret_to_update = await self.get_secret(user_id, secret_id)
        if not secret_to_update:
             logger.warning(f"Attempt to update non-existent or unauthorized secret {secret_id} by user {user_id}")
             return False

        # Prevent changing user_id or id via updates
        if "user_id" in updates: del updates["user_id"]
        if "id" in updates: del updates["id"]
        updates["updated_at"] = datetime.now(timezone.utc) # Ensure timestamp update

        try:
            if self.db:
                doc_ref = self.db.collection("secrets").document(secret_id)
                await asyncio.to_thread(doc_ref.update, updates)
            else: # In-memory
                self.in_memory_db["secrets"][secret_id].update(updates)
            return True
        except Exception as e:
             logger.error(f"DB Error update_secret {secret_id}: {e}", exc_info=True)
             return False

    async def delete_secret(self, user_id: str, secret_id: str) -> bool:
        # Perform ownership check before deleting
        secret_to_delete = await self.get_secret(user_id, secret_id)
        if not secret_to_delete:
            logger.warning(f"Attempt to delete non-existent or unauthorized secret {secret_id} by user {user_id}")
            return False # Not found or not owned

        try:
            if self.db:
                await asyncio.to_thread(self.db.collection("secrets").document(secret_id).delete)
            else: # In-memory
                self.in_memory_db["secrets"].pop(secret_id, None)
            logger.info(f"Deleted secret {secret_id} for user {user_id}")
            return True
        except Exception as e:
             logger.error(f"DB Error delete_secret {secret_id}: {e}", exc_info=True)
             return False

# --- Singleton Instance ---
_db_manager: Optional[DatabaseManager] = None
def get_db_manager() -> DatabaseManager:
    """Gets the singleton DatabaseManager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager