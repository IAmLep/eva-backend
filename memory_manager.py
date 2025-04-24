"""
Memory Manager for EVA backend.

Implements a multi-tiered memory system for storing and retrieving
user memories (core facts, conversation context, events). Integrates
with the database manager for persistence.
"""

import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from pydantic import BaseModel, Field

# --- Local Imports ---
from config import settings
from database import get_db_manager # Assuming database.py provides this interface
from exceptions import DatabaseError, NotFoundException, AuthorizationError
# Import models used by this manager
from models import User, Memory, MemorySource, MemoryCategory

# Logger configuration
logger = logging.getLogger(__name__)

# --- Enums (Redefine locally or import if models.py changes often) ---
# It's often safer to redefine enums if they are critical to this module's logic
# and models.py might change, or ensure models.py is stable.
class MemoryType(str, Enum):
    CORE = "core"
    CONVERSATIONAL = "conversational"
    EVENT = "event"
    SYSTEM = "system"

# MemoryCategory already imported from models

# --- Helper Models ---
class MemoryRelevance(BaseModel):
    """Model for memory relevance scoring results."""
    memory_id: str
    score: float
    match_reason: str


class MemoryManager:
    """
    Handles multi-tiered memory storage and retrieval operations.

    Provides methods for creating, reading, updating, deleting, and
    finding relevant memories across different types (core, conversational, event).
    """

    def __init__(self):
        """Initialize memory manager with database connection and settings."""
        self.settings = settings
        self.db = get_db_manager() # Get the singleton DB manager instance

        # Basic in-memory stats (could be expanded)
        self.memory_stats: Dict[MemoryType, Dict[str, Any]] = {
            mem_type: {"count": 0, "last_updated": None} for mem_type in MemoryType
        }
        # TODO: Consider periodically loading stats from DB on startup

        logger.info("Memory manager initialized")

    # --- Memory Creation Methods ---

    async def create_core_memory(
        self,
        user_id: str,
        content: str,
        category: MemoryCategory, # Use the enum directly
        entity: Optional[str] = None,
        importance: int = 5,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """Creates a new core (long-term) memory."""
        if not content:
            raise ValueError("Memory content cannot be empty")
        if not 1 <= importance <= 10:
             logger.warning(f"Importance {importance} out of range (1-10), clamping.")
             importance = max(1, min(10, importance))

        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Prepare metadata, ensuring category and importance are stored
        final_metadata = metadata or {}
        final_metadata.update({
            "category": category.value,
            "importance": importance,
            "entity": entity, # Store entity if provided
        })
        # Remove None values from metadata if necessary
        final_metadata = {k: v for k, v in final_metadata.items() if v is not None}

        # Prepare tags, ensuring category is included
        final_tags = tags or []
        if category.value not in final_tags:
            final_tags.append(category.value)
        if entity and entity not in final_tags:
             final_tags.append(entity) # Add entity as a tag

        memory = Memory(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            source=MemoryType.CORE.value,
            metadata=final_metadata,
            tags=list(set(final_tags)), # Ensure unique tags
            created_at=now,
            updated_at=now,
            is_synced=False, # New memories are not synced initially
            importance=importance # Also store top-level importance if needed
        )

        try:
            await self.db.create_memory(memory) # Use the method from database manager
            self._update_stats(MemoryType.CORE, 1)
            logger.info(f"Created core memory {memory_id} for user {user_id} (Category: {category.value})")
            return memory
        except Exception as e:
            logger.error(f"Error creating core memory in DB: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create core memory: {e}")

    async def create_event_memory(
        self,
        user_id: str,
        content: str,
        event_time: datetime,
        expiration: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """Creates a new event memory (e.g., reminder, appointment)."""
        if not content:
            raise ValueError("Event memory content cannot be empty")

        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Default expiration: event_time + 1 day
        final_expiration = expiration or (event_time + timedelta(days=1))

        final_metadata = metadata or {}
        final_metadata.update({
            "event_time": event_time.isoformat(),
            "expiration": final_expiration.isoformat(),
            "completed": False, # Events start as not completed
        })
        final_metadata = {k: v for k, v in final_metadata.items() if v is not None}

        final_tags = tags or []
        if "event" not in final_tags:
            final_tags.append("event")

        memory = Memory(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            source=MemoryType.EVENT.value,
            metadata=final_metadata,
            tags=list(set(final_tags)),
            created_at=now,
            updated_at=now,
            is_synced=False,
            importance=8, # Events are generally important
            expiration=final_expiration # Store top-level expiration as well
        )

        try:
            await self.db.create_memory(memory)
            self._update_stats(MemoryType.EVENT, 1)
            logger.info(f"Created event memory {memory_id} for user {user_id} at {event_time.isoformat()}")
            return memory
        except Exception as e:
            logger.error(f"Error creating event memory in DB: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create event memory: {e}")

    async def create_conversation_memory(
        self,
        user_id: str,
        content: str,
        conversation_id: str,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Memory:
        """Creates a new conversation memory (e.g., summary, key point)."""
        if not content:
            raise ValueError("Conversation memory content cannot be empty")

        memory_id = str(uuid.uuid4())
        now = datetime.utcnow()

        final_metadata = metadata or {}
        final_metadata.update({
            "conversation_id": conversation_id,
            "entities": entities or [],
        })
        final_metadata = {k: v for k, v in final_metadata.items() if v is not None}

        final_tags = tags or []
        if "conversation" not in final_tags:
            final_tags.append("conversation")
        if entities:
             final_tags.extend(entities)

        memory = Memory(
            memory_id=memory_id,
            user_id=user_id,
            content=content,
            source=MemoryType.CONVERSATIONAL.value,
            metadata=final_metadata,
            tags=list(set(final_tags)),
            created_at=now,
            updated_at=now,
            is_synced=False,
            importance=3, # Conversational snippets usually less critical than core/event
        )

        try:
            await self.db.create_memory(memory)
            self._update_stats(MemoryType.CONVERSATIONAL, 1)
            logger.info(f"Created conversation memory {memory_id} for user {user_id}")
            return memory
        except Exception as e:
            logger.error(f"Error creating conversation memory in DB: {e}", exc_info=True)
            raise DatabaseError(f"Failed to create conversation memory: {e}")

    # --- Memory Retrieval Methods ---

    async def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """Gets a specific memory by ID, checking ownership."""
        try:
            memory = await self.db.get_memory(memory_id)
            if not memory:
                # logger.info(f"Memory {memory_id} not found.") # Less noisy
                return None

            if memory.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access memory {memory_id} owned by {memory.user_id}")
                raise AuthorizationError("Not authorized to access this memory")

            return memory
        except AuthorizationError:
             raise # Re-raise auth errors directly
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}", exc_info=True)
            # Don't raise DatabaseError here, let caller handle None or raise specific error if needed
            return None # Indicate failure to retrieve

    async def get_memories_by_type(
        self,
        user_id: str,
        memory_type: MemoryType,
        limit: int = 50,
        offset: int = 0,
        filter_tags: Optional[List[str]] = None
    ) -> List[Memory]:
        """Gets memories of a specific type for a user."""
        try:
            # Database manager should handle filtering by type and tags if possible
            memories = await self.db.get_user_memories(
                user_id=user_id,
                limit=limit,
                offset=offset,
                memory_type=memory_type.value, # Pass type to DB layer
                tags=filter_tags
            )
            return memories
        except Exception as e:
            logger.error(f"Error retrieving memories by type {memory_type.value}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve {memory_type.value} memories: {e}")

    async def get_core_memories(
        self,
        user_id: str,
        category: Optional[MemoryCategory] = None,
        entity: Optional[str] = None,
        limit: int = 50
    ) -> List[Memory]:
        """Gets core memories, optionally filtered."""
        tags_to_filter = []
        if category:
            tags_to_filter.append(category.value)
        if entity:
            tags_to_filter.append(entity)

        # Rely on get_memories_by_type with tag filtering
        return await self.get_memories_by_type(
            user_id=user_id,
            memory_type=MemoryType.CORE,
            limit=limit,
            filter_tags=tags_to_filter if tags_to_filter else None
        )

    async def get_event_memories(
        self,
        user_id: str,
        include_past: bool = False,
        days_ahead: int = 7,
        limit: int = 10
    ) -> List[Memory]:
        """Gets upcoming (or past) event memories."""
        try:
            now = datetime.utcnow()
            start_time = now - timedelta(days=365*10) if include_past else now # Wide past window if included
            end_time = now + timedelta(days=days_ahead)

            # Assuming database layer can filter by event time range
            # If not, retrieve all events and filter here.
            events = await self.db.get_events_by_timerange(user_id, start_time, end_time)

            # Filter out completed events
            active_events = [
                event for event in events
                if not event.metadata.get("completed", False)
            ]

            # Sort by event time (ascending)
            active_events.sort(
                key=lambda m: m.metadata.get("event_time", "9999") # Sort by iso string
            )

            return active_events[:limit]
        except Exception as e:
            logger.error(f"Error retrieving event memories: {e}", exc_info=True)
            raise DatabaseError(f"Failed to retrieve event memories: {e}")

    async def get_relevant_memories(
        self,
        user_id: str,
        query: str,
        entities: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Tuple[Memory, float]]:
        """
        Retrieves memories relevant to a query and/or entities.

        Uses a basic keyword/entity matching score. Embeddings would be better.
        """
        if not query and not entities:
            return [] # Need something to search for

        try:
            # --- Simple Keyword/Entity Scoring ---
            # In production, replace this with vector search/embeddings

            # 1. Fetch potentially relevant memories (e.g., recent, core, matching tags)
            # This part needs optimization. Fetching ALL memories is inefficient.
            # Let's try fetching core + recent conversational/event + tagged by entity
            potential_memories = set()

            # Core memories are generally relevant
            core_mems = await self.get_core_memories(user_id, limit=self.settings.MEMORY_MAX_CORE_MEMORIES)
            potential_memories.update(core_mems)

            # Recent conversational/event memories
            recent_conv = await self.get_memories_by_type(user_id, MemoryType.CONVERSATIONAL, limit=20)
            potential_memories.update(recent_conv)
            recent_events = await self.get_event_memories(user_id, include_past=True, days_ahead=1, limit=self.settings.MEMORY_MAX_EVENT_MEMORIES)
            potential_memories.update(recent_events)

            # Memories tagged with entities
            if entities:
                 entity_mems = await self.db.get_user_memories(user_id, limit=50, tags=entities)
                 potential_memories.update(entity_mems)

            logger.debug(f"Found {len(potential_memories)} potential memories for relevance scoring.")

            # 2. Score the potential memories
            scored_memories: List[Tuple[Memory, float, str]] = []
            query_terms = set(term for term in query.lower().split() if len(term) > 2) # Simple tokenization
            entity_set = set(e.lower() for e in entities) if entities else set()

            for memory in potential_memories:
                score = 0.0
                match_reason = []
                content_lower = memory.content.lower()
                tags_lower = set(t.lower() for t in memory.tags)
                meta_entity = memory.metadata.get("entity", "").lower()

                # Query term matching
                terms_found = query_terms.intersection(content_lower.split())
                if terms_found:
                    score += 0.5 * len(terms_found) # Score based on number of terms found
                    match_reason.append(f"Query terms: {', '.join(terms_found)}")

                # Entity matching (content, metadata, tags)
                entities_found_meta = entity_set.intersection({meta_entity} if meta_entity else set())
                entities_found_tags = entity_set.intersection(tags_lower)
                entities_found_content = {e for e in entity_set if e in content_lower}

                if entities_found_meta:
                    score += 1.5 * len(entities_found_meta) # High score for direct meta match
                    match_reason.append(f"Meta entity: {', '.join(entities_found_meta)}")
                if entities_found_tags:
                    score += 1.0 * len(entities_found_tags) # Good score for tag match
                    match_reason.append(f"Tags: {', '.join(entities_found_tags)}")
                # Lower score for just content match to avoid noise
                content_matches = entities_found_content - entities_found_meta - entities_found_tags
                if content_matches:
                     score += 0.3 * len(content_matches)
                     match_reason.append(f"Content: {', '.join(content_matches)}")


                # Boost based on memory type and importance
                base_importance = memory.metadata.get("importance", 5)
                if memory.source == MemoryType.CORE.value:
                    score *= (1.2 + base_importance / 10.0) # Boost core memories significantly
                elif memory.source == MemoryType.EVENT.value:
                     # Boost recent/upcoming events
                     event_time_str = memory.metadata.get("event_time")
                     try:
                          event_dt = datetime.fromisoformat(event_time_str)
                          time_diff_hours = abs((event_dt - datetime.utcnow()).total_seconds() / 3600)
                          if time_diff_hours < 24: score *= 1.5 # Boost events within 24h
                     except: pass
                     score *= (1.1 + base_importance / 10.0)
                else: # Conversational
                     score *= (0.8 + base_importance / 10.0)

                # Add recency boost (newer memories are slightly more relevant)
                time_since_update = (datetime.utcnow() - memory.updated_at).total_seconds()
                recency_boost = max(0, 1 - (time_since_update / (86400 * 30))) # Linear decay over 30 days
                score *= (1 + 0.2 * recency_boost)


                # Only include memories with a minimum score
                min_score_threshold = 0.2
                if score >= min_score_threshold:
                    scored_memories.append((memory, score, ", ".join(match_reason)))

            # Sort by score (desc) and return top N
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [(memory, score) for memory, score, reason in scored_memories[:limit]]

            if top_memories:
                 log_details = [(m.memory_id, f"{s:.2f}", r) for m, s, r in scored_memories[:limit]]
                 logger.info(f"Found {len(top_memories)} relevant memories. Top: {log_details}")
            else:
                 logger.info("No relevant memories found matching criteria.")

            return top_memories

        except Exception as e:
            logger.error(f"Error finding relevant memories: {e}", exc_info=True)
            # Return empty list on error rather than raising DatabaseError unless critical
            return []

    # --- Memory Modification Methods ---

    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Updates an existing memory, checking ownership."""
        try:
            # Get current memory to check ownership and merge metadata/tags
            memory = await self.get_memory(memory_id, user_id) # Handles ownership check
            if not memory:
                raise NotFoundException(f"Memory {memory_id} not found")

            update_data = {}
            has_changes = False

            # Merge metadata if provided
            if "metadata" in updates and isinstance(updates["metadata"], dict):
                # Perform a deep merge if necessary, or simple update
                new_metadata = {**memory.metadata, **updates["metadata"]}
                if new_metadata != memory.metadata:
                    update_data["metadata"] = new_metadata
                    has_changes = True
            elif "metadata" in updates:
                 logger.warning(f"Invalid metadata format in update for memory {memory_id}. Ignored.")

            # Replace tags if provided
            if "tags" in updates and isinstance(updates["tags"], list):
                 new_tags = list(set(updates["tags"])) # Ensure unique
                 if set(new_tags) != set(memory.tags):
                    update_data["tags"] = new_tags
                    has_changes = True
            elif "tags" in updates:
                 logger.warning(f"Invalid tags format in update for memory {memory_id}. Ignored.")

            # Update other fields directly
            for field in ["content", "importance", "expiration", "source"]:
                if field in updates and updates[field] != getattr(memory, field, None):
                    update_data[field] = updates[field]
                    has_changes = True
                    # Special handling for importance if needed (e.g., update metadata too)
                    if field == "importance" and "metadata" not in update_data:
                         update_data["metadata"] = memory.metadata
                    if field == "importance" and "metadata" in update_data:
                         update_data["metadata"]["importance"] = updates[field]
                    # Similar handling for expiration
                    if field == "expiration" and "metadata" not in update_data:
                        update_data["metadata"] = memory.metadata
                    if field == "expiration" and "metadata" in update_data:
                        update_data["metadata"]["expiration"] = updates[field].isoformat() if updates[field] else None


            if not has_changes:
                 logger.info(f"No changes detected for memory {memory_id}. Update skipped.")
                 return True # No changes, but considered successful

            # Always update timestamp and sync status
            update_data["updated_at"] = datetime.utcnow()
            update_data["is_synced"] = False

            success = await self.db.update_memory(memory_id, update_data)
            if success:
                logger.info(f"Updated memory {memory_id} for user {user_id}")
            else:
                 logger.error(f"Database update failed for memory {memory_id}")

            return success

        except (NotFoundException, AuthorizationError):
            raise # Re-raise specific errors
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to update memory: {e}")

    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """Deletes a memory, checking ownership."""
        try:
            # Ownership check is implicitly done by get_memory
            memory = await self.get_memory(memory_id, user_id)
            if not memory:
                # Already logged by get_memory if not found, or auth error raised
                raise NotFoundException(f"Memory {memory_id} not found or not accessible.")

            success = await self.db.delete_memory(user_id, memory_id) # Pass user_id for DB layer checks too

            if success:
                logger.info(f"Deleted memory {memory_id} for user {user_id}")
                # Update stats (decrement count)
                try:
                     mem_type = MemoryType(memory.source)
                     self._update_stats(mem_type, -1)
                except ValueError:
                     logger.warning(f"Memory {memory_id} had unknown source type '{memory.source}' during deletion stat update.")
            else:
                 logger.error(f"Database delete failed for memory {memory_id}")

            return success
        except (NotFoundException, AuthorizationError):
            raise # Re-raise specific errors
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to delete memory: {e}")

    async def complete_event(self, memory_id: str, user_id: str) -> bool:
        """Marks an event memory as completed."""
        try:
            memory = await self.get_memory(memory_id, user_id) # Checks ownership
            if not memory:
                raise NotFoundException(f"Event memory {memory_id} not found")

            if memory.source != MemoryType.EVENT.value:
                logger.warning(f"Attempted to complete non-event memory {memory_id} (type: {memory.source})")
                # Consider raising a specific error or just returning False
                raise ValueError(f"Memory {memory_id} is not an event memory.")

            if memory.metadata.get("completed", False):
                logger.info(f"Event memory {memory_id} is already marked as completed.")
                return True # Already done, success

            # Prepare update to set completed = True in metadata
            updates = {
                "metadata": {**memory.metadata, "completed": True}
            }

            return await self.update_memory(memory_id, user_id, updates)

        except (NotFoundException, AuthorizationError, ValueError):
            raise # Re-raise specific errors
        except Exception as e:
            logger.error(f"Error completing event {memory_id}: {e}", exc_info=True)
            raise DatabaseError(f"Failed to complete event: {e}")

    # --- Utility Methods ---

    def _update_stats(self, memory_type: MemoryType, delta: int):
        """Updates internal memory statistics."""
        if memory_type in self.memory_stats:
            self.memory_stats[memory_type]["count"] = max(0, self.memory_stats[memory_type].get("count", 0) + delta)
            self.memory_stats[memory_type]["last_updated"] = datetime.utcnow()
        else:
             logger.warning(f"Attempted to update stats for unknown memory type: {memory_type}")

    async def get_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """Retrieves memory statistics for a user (counts from DB)."""
        # This should ideally query the DB for accurate counts per type
        # Placeholder implementation using internal stats (less accurate)
        logger.warning("get_memory_statistics currently uses potentially inaccurate in-memory counts.")
        # TODO: Implement DB query for accurate counts per user/type
        return {
            mem_type.value: stats
            for mem_type, stats in self.memory_stats.items()
        }


# --- Singleton Instance ---
_memory_manager: Optional[MemoryManager] = None

def get_memory_manager() -> MemoryManager:
    """Gets the singleton MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager