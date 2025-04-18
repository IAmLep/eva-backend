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
    try: from google.cloud.firestore_v1.base_query import FieldFilter
    except ImportError: from google.cloud.firestore_v1.query import FieldFilter
    from google.api_core import exceptions as google_exceptions

    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    # Define dummy types
    # ... (dummy definitions remain the same) ...

# --- Local Imports ---
from config import get_settings
from models import Memory, User, UserInDB, Conversation, SyncState, ApiKey, UserRateLimit, MemorySource, MemoryCategory # Added UserInDB
from exceptions import DatabaseError, ConfigurationError, NotFoundException # Import NotFoundException

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        # ... (Initialization remains the same) ...
        pass

    def _get_collection(self, name: str): ... # (Remains the same)

    # --- User Operations ---
    async def create_user(self, user: UserInDB) -> str: # Expect UserInDB now
        user_id = user.id or str(uuid.uuid4())
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
            else:
                # ... (in-memory logic) ...
                # Ensure return type matches UserInDB
                pass
        except Exception as e:
            # ... (error logging) ...
            return None

    async def get_user_by_field(self, field_name: str, value: Any) -> Optional[UserInDB]: # Return UserInDB
         try:
             if self.db:
                  # ... (Firestore query) ...
                  for doc in results:
                       user_data = doc.to_dict()
                       user_data["id"] = doc.id
                       return UserInDB(**user_data) # Cast to UserInDB
                  return None
             else:
                  # ... (in-memory logic) ...
                  # Ensure return type matches UserInDB
                  pass
         except Exception as e:
              # ... (error logging) ...
              return None

    async def get_user_by_username(self, username: str) -> Optional[UserInDB]: # Return UserInDB
        return await self.get_user_by_field("username", username)

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]: # Return UserInDB
        return await self.get_user_by_field("email", email)

    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool: ... # (Remains the same)
    async def delete_user(self, user_id: str) -> bool: ... # (Remains the same)

    # --- Memory Operations ---
    async def create_memory(self, memory: Memory) -> bool: ... # (Remains the same)
    async def get_memory(self, memory_id: str) -> Optional[Memory]: ... # (Remains the same)
    async def get_user_memories(self, user_id: str, limit: int = 50, offset: int = 0, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Memory]: ... # (Remains the same)
    async def update_memory(self, memory_id: str, updates: Dict[str, Any]) -> bool: ... # (Remains the same)
    async def delete_memory(self, user_id: str, memory_id: str) -> bool: ... # (Remains the same)
    async def get_events_by_timerange(self, user_id: str, start_time: datetime, end_time: datetime) -> List[Memory]: ... # (Remains the same)

    async def get_memories_by_query(self, user_id: str, query: str, limit: int = 10) -> List[Memory]:
        """Placeholder: Retrieves memories based on a simple text query."""
        # In a real implementation, this would use vector search or more advanced text matching.
        # Simple keyword matching for demonstration:
        logger.warning("get_memories_by_query using simple keyword matching.")
        query_terms = set(q.lower() for q in query.split() if len(q) > 2)
        all_memories = await self.get_user_memories(user_id, limit=200) # Fetch more to filter
        matched = []
        for mem in all_memories:
            content_lower = mem.content.lower()
            if any(term in content_lower for term in query_terms):
                matched.append(mem)
                if len(matched) >= limit:
                    break
        return matched

    async def cleanup_old_memories(self, user_id: str, days_threshold: int) -> int:
        """Placeholder: Deletes memories older than the specified threshold."""
        # This requires careful implementation to avoid accidental data loss.
        # Consider only deleting certain types (e.g., conversational) or adding archiving.
        logger.warning(f"Placeholder cleanup_old_memories called for user {user_id}, threshold {days_threshold} days. No deletion performed.")
        # Example logic (needs refinement):
        # threshold_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
        # query = self.db.collection("memories").where("user_id", "==", user_id).where("updated_at", "<", threshold_date)
        # ... iterate and delete ...
        return 0 # Return count of deleted items

    async def cleanup_duplicate_memories(self, user_id: str) -> int:
        """Placeholder: Deletes duplicate memories based on content hash or similarity."""
        logger.warning(f"Placeholder cleanup_duplicate_memories called for user {user_id}. No deletion performed.")
        # Logic would involve fetching memories, calculating hashes/embeddings, identifying duplicates, and deleting.
        return 0 # Return count of deleted items

    # --- Conversation Operations (Stubs) ---
    # ... (Stubs remain the same) ...

    # --- Sync Operations (Stubs) ---
    # ... (Stubs remain the same) ...

    # --- API Key Operations ---
    async def get_api_key_by_hash(self, hashed_key: str) -> Optional[ApiKey]: ... # (Remains the same, used by old security logic)

    async def get_api_keys_by_prefix(self, prefix: str) -> List[ApiKey]:
        """Gets API keys matching a given prefix."""
        # This query requires an index on 'prefix'.
        keys = []
        try:
            if self.db:
                if not FieldFilter: raise NotImplementedError("Firestore FieldFilter not available.")
                query = self.db.collection("api_keys").where(filter=FieldFilter("prefix", "==", prefix))
                results = await asyncio.to_thread(query.stream)
                for doc in results:
                    key_data = doc.to_dict()
                    key_data["key_id"] = doc.id
                    keys.append(ApiKey(**key_data))
            else: # In-memory
                for key_id, key_data in self.in_memory_db["api_keys"].items():
                    if key_data.get("prefix") == prefix:
                        key_data["key_id"] = key_id
                        keys.append(ApiKey(**key_data))
            logger.debug(f"Retrieved {len(keys)} API keys with prefix '{prefix}'")
            return keys
        except Exception as e:
            logger.error(f"Error retrieving API keys by prefix '{prefix}': {e}", exc_info=True)
            return [] # Return empty list on error


    async def update_api_key_usage(self, key_id: str) -> bool: ... # (Remains the same)

    # --- Secret Category & Secret Operations ---
    # (Implementations remain the same, relying on database methods)
    async def create_category(self, category_data: Dict[str, Any]) -> bool: ...
    async def get_category(self, user_id: str, category_id: str) -> Optional[Dict[str, Any]]: ...
    async def get_user_categories(self, user_id: str) -> List[Dict[str, Any]]: ...
    async def delete_category(self, category_id: str) -> bool: ...
    async def create_secret(self, secret_data: Dict[str, Any]) -> bool: ...
    async def get_secret(self, secret_id: str) -> Optional[Dict[str, Any]]: ...
    async def get_user_secrets(self, user_id: str, category_id: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]: ...
    async def update_secret(self, secret_id: str, updates: Dict[str, Any]) -> bool: ...
    async def delete_secret(self, secret_id: str) -> bool: ...

# --- Singleton Instance ---
_db_manager: Optional[DatabaseManager] = None
def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None: _db_manager = DatabaseManager()
    return _db_manager