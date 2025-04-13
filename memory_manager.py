"""
Memory Manager for EVA backend.

This module implements a sophisticated multi-tiered memory system for
storing and retrieving different types of user memories, including core facts,
conversation context, and time-based events.

Current Date: 2025-04-12
Current User: IAmLep
"""

import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Set

from pydantic import BaseModel, Field

from config import get_settings
from database import get_db_manager
from exceptions import DatabaseError, NotFoundException
from models import User, Memory

# Logger configuration
logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """
    Memory type enumeration.
    
    Attributes:
        CORE: Long-term important memories (people, places, preferences)
        CONVERSATIONAL: Recent conversation memories
        EVENT: Time-based memories like appointments
    """
    CORE = "core"
    CONVERSATIONAL = "conversational"
    EVENT = "event"


class MemoryCategory(str, Enum):
    """
    Memory category enumeration for core memories.
    
    Attributes:
        PERSON: Information about people
        PLACE: Information about places
        PREFERENCE: User preferences and likes/dislikes
        FACT: General factual information
        OTHER: Miscellaneous information
    """
    PERSON = "person"
    PLACE = "place"
    PREFERENCE = "preference"
    FACT = "fact"
    OTHER = "other"


class MemoryCommand(BaseModel):
    """
    Memory command model for extracting memory management commands.
    
    Attributes:
        command_type: Type of memory command
        content: Main memory content
        entity: Optional primary entity for the memory
        category: Optional memory category
        event_time: Optional time for event memories
        expiration: Optional expiration time
    """
    command_type: str  # "remember", "forget", "remind"
    content: str
    entity: Optional[str] = None
    category: Optional[MemoryCategory] = None
    event_time: Optional[datetime] = None
    expiration: Optional[datetime] = None


class MemoryRelevance(BaseModel):
    """
    Memory relevance scoring model.
    
    Attributes:
        memory_id: Memory identifier
        score: Relevance score (0-1)
        match_reason: Reason for the relevance match
    """
    memory_id: str
    score: float
    match_reason: str


class MemoryManager:
    """
    Memory Manager for handling multi-tiered memory storage and retrieval.
    
    This class provides methods for creating, retrieving, updating, and
    deleting memories across different memory types (core, conversational, event).
    """
    
    def __init__(self):
        """Initialize memory manager with database connection."""
        self.settings = get_settings()
        self.db = get_db_manager()
        
        # Initialize memory statistics
        self.memory_stats = {
            MemoryType.CORE: {"count": 0, "last_updated": None},
            MemoryType.CONVERSATIONAL: {"count": 0, "last_updated": None},
            MemoryType.EVENT: {"count": 0, "last_updated": None}
        }
        
        logger.info("Memory manager initialized")
    
    async def create_core_memory(
        self,
        user_id: str,
        content: str,
        category: Union[MemoryCategory, str],
        entity: Optional[str] = None,
        importance: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Create a new core memory.
        
        Args:
            user_id: User ID who owns the memory
            content: Memory content
            category: Memory category
            entity: Optional primary entity for the memory
            importance: Importance score (1-10)
            metadata: Additional metadata
            
        Returns:
            Memory: Created memory object
            
        Raises:
            DatabaseError: If memory creation fails
        """
        try:
            # Standardize category
            if isinstance(category, str):
                category = MemoryCategory(category.lower())
                
            # Generate memory ID
            memory_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            metadata.update({
                "category": category.value,
                "importance": importance
            })
            
            if entity:
                metadata["entity"] = entity
            
            # Create memory object
            memory = Memory(
                memory_id=memory_id,
                user_id=user_id,
                content=content,
                source=MemoryType.CORE.value,
                metadata=metadata,
                tags=[category.value],
                created_at=now,
                updated_at=now,
                is_synced=False
            )
            
            # Store in database
            await self.db.create_memory(memory)
            
            # Update stats
            self.memory_stats[MemoryType.CORE]["count"] += 1
            self.memory_stats[MemoryType.CORE]["last_updated"] = now
            
            logger.info(f"Created core memory {memory_id} for user {user_id}: {category.value}")
            return memory
            
        except Exception as e:
            logger.error(f"Error creating core memory: {str(e)}")
            raise DatabaseError(f"Failed to create core memory: {str(e)}")
    
    async def create_event_memory(
        self,
        user_id: str,
        content: str,
        event_time: datetime,
        expiration: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Create a new event memory (appointment, reminder, etc).
        
        Args:
            user_id: User ID who owns the memory
            content: Memory content (what to remember)
            event_time: When the event occurs
            expiration: When the memory should expire (default: event_time + 1 day)
            metadata: Additional metadata
            
        Returns:
            Memory: Created memory object
            
        Raises:
            DatabaseError: If memory creation fails
        """
        try:
            # Generate memory ID
            memory_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Set default expiration if not provided
            if expiration is None:
                expiration = event_time + timedelta(days=1)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            metadata.update({
                "event_time": event_time.isoformat(),
                "expiration": expiration.isoformat(),
                "completed": False
            })
            
            # Create memory object
            memory = Memory(
                memory_id=memory_id,
                user_id=user_id,
                content=content,
                source=MemoryType.EVENT.value,
                metadata=metadata,
                tags=["event"],
                created_at=now,
                updated_at=now,
                is_synced=False
            )
            
            # Store in database
            await self.db.create_memory(memory)
            
            # Update stats
            self.memory_stats[MemoryType.EVENT]["count"] += 1
            self.memory_stats[MemoryType.EVENT]["last_updated"] = now
            
            logger.info(f"Created event memory {memory_id} for user {user_id} at {event_time.isoformat()}")
            return memory
            
        except Exception as e:
            logger.error(f"Error creating event memory: {str(e)}")
            raise DatabaseError(f"Failed to create event memory: {str(e)}")
    
    async def create_conversation_memory(
        self,
        user_id: str,
        content: str,
        conversation_id: str,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Create a new conversation memory (summary or important point).
        
        Args:
            user_id: User ID who owns the memory
            content: Memory content (conversation summary or highlight)
            conversation_id: ID of the conversation
            entities: List of entities mentioned
            metadata: Additional metadata
            
        Returns:
            Memory: Created memory object
            
        Raises:
            DatabaseError: If memory creation fails
        """
        try:
            # Generate memory ID
            memory_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            metadata.update({
                "conversation_id": conversation_id
            })
            
            if entities:
                metadata["entities"] = entities
            
            # Create memory object
            memory = Memory(
                memory_id=memory_id,
                user_id=user_id,
                content=content,
                source=MemoryType.CONVERSATIONAL.value,
                metadata=metadata,
                tags=["conversation"] + (entities or []),
                created_at=now,
                updated_at=now,
                is_synced=False
            )
            
            # Store in database
            await self.db.create_memory(memory)
            
            # Update stats
            self.memory_stats[MemoryType.CONVERSATIONAL]["count"] += 1
            self.memory_stats[MemoryType.CONVERSATIONAL]["last_updated"] = now
            
            logger.info(f"Created conversation memory {memory_id} for user {user_id}")
            return memory
            
        except Exception as e:
            logger.error(f"Error creating conversation memory: {str(e)}")
            raise DatabaseError(f"Failed to create conversation memory: {str(e)}")
    
    async def get_memory(self, memory_id: str, user_id: str) -> Optional[Memory]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            user_id: User ID for authorization
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
            
        Raises:
            AuthorizationError: If user doesn't own the memory
        """
        try:
            # Get memory from database
            memory = await self.db.get_memory(memory_id)
            
            if not memory:
                return None
                
            # Check ownership
            if memory.user_id != user_id:
                logger.warning(f"User {user_id} attempted to access memory {memory_id} owned by {memory.user_id}")
                from exceptions import AuthorizationError
                raise AuthorizationError("Not authorized to access this memory")
            
            return memory
            
        except Exception as e:
            if isinstance(e, AuthorizationError):
                raise
                
            logger.error(f"Error retrieving memory {memory_id}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve memory: {str(e)}")
    
    async def get_memories_by_type(
        self,
        user_id: str,
        memory_type: MemoryType,
        limit: int = 50,
        offset: int = 0,
        filter_tags: Optional[List[str]] = None
    ) -> List[Memory]:
        """
        Get memories by type.
        
        Args:
            user_id: User ID to get memories for
            memory_type: Type of memories to retrieve
            limit: Maximum number of memories to return
            offset: Offset for pagination
            filter_tags: Optional tags to filter by
            
        Returns:
            List[Memory]: List of memories
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            # Get memories from database
            memories = await self.db.get_user_memories(
                user_id=user_id,
                limit=limit,
                offset=offset,
                tags=filter_tags
            )
            
            # Filter by source/type
            typed_memories = [
                memory for memory in memories
                if memory.source == memory_type.value
            ]
            
            return typed_memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories by type {memory_type.value}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve memories: {str(e)}")
    
    async def get_core_memories(
        self,
        user_id: str,
        category: Optional[MemoryCategory] = None,
        entity: Optional[str] = None,
        limit: int = 50
    ) -> List[Memory]:
        """
        Get core memories, optionally filtered by category or entity.
        
        Args:
            user_id: User ID to get memories for
            category: Optional category filter
            entity: Optional entity filter
            limit: Maximum number of memories to return
            
        Returns:
            List[Memory]: List of core memories
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            # Start with all core memories
            memories = await self.get_memories_by_type(
                user_id=user_id,
                memory_type=MemoryType.CORE,
                limit=limit
            )
            
            # Filter by category if provided
            if category:
                memories = [
                    memory for memory in memories
                    if memory.metadata.get("category") == category.value
                ]
            
            # Filter by entity if provided
            if entity:
                memories = [
                    memory for memory in memories
                    if memory.metadata.get("entity") == entity
                ]
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving core memories: {str(e)}")
            raise DatabaseError(f"Failed to retrieve core memories: {str(e)}")
    
    async def get_event_memories(
        self,
        user_id: str,
        include_past: bool = False,
        days_ahead: int = 7,
        limit: int = 10
    ) -> List[Memory]:
        """
        Get upcoming event memories.
        
        Args:
            user_id: User ID to get memories for
            include_past: Whether to include past events
            days_ahead: How many days ahead to look
            limit: Maximum number of events to return
            
        Returns:
            List[Memory]: List of event memories
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            # Get all event memories
            all_events = await self.get_memories_by_type(
                user_id=user_id,
                memory_type=MemoryType.EVENT,
                limit=100  # Get more since we'll filter
            )
            
            now = datetime.utcnow()
            future_limit = now + timedelta(days=days_ahead)
            
            # Filter events
            filtered_events = []
            for memory in all_events:
                # Parse event_time
                event_time_str = memory.metadata.get("event_time")
                if not event_time_str:
                    continue
                    
                try:
                    event_time = datetime.fromisoformat(event_time_str)
                except ValueError:
                    continue
                
                # Check if event is completed
                if memory.metadata.get("completed", False):
                    continue
                    
                # Filter based on time
                if include_past or event_time >= now:
                    if event_time <= future_limit:
                        filtered_events.append(memory)
            
            # Sort by event_time
            filtered_events.sort(
                key=lambda m: datetime.fromisoformat(m.metadata.get("event_time", "9999"))
            )
            
            # Return limited results
            return filtered_events[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving event memories: {str(e)}")
            raise DatabaseError(f"Failed to retrieve event memories: {str(e)}")
    
    async def get_relevant_memories(
        self,
        user_id: str,
        query: str,
        entities: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Tuple[Memory, float]]:
        """
        Get memories relevant to a query or entities.
        
        Args:
            user_id: User ID to get memories for
            query: Query text to match against memories
            entities: List of entities to find memories for
            limit: Maximum number of memories to return
            
        Returns:
            List[Tuple[Memory, float]]: List of memories with relevance scores
            
        Raises:
            DatabaseError: If retrieval fails
        """
        try:
            # For now, we'll use a simple keyword matching approach
            # In a real implementation, you might use embeddings and vector search
            
            # Get all user memories
            # In a production system, this would be optimized to not load all memories
            all_memories = await self.db.get_user_memories(user_id, limit=200)
            
            # Score memories by relevance
            scored_memories = []
            
            for memory in all_memories:
                score = 0.0
                match_reason = ""
                
                # Check for query terms in content
                query_terms = query.lower().split()
                content_lower = memory.content.lower()
                
                for term in query_terms:
                    if term in content_lower:
                        score += 0.2  # Base score for term match
                        # Bonus for term frequency
                        score += 0.05 * content_lower.count(term)
                        match_reason = f"Contains term: {term}"
                
                # Check for entity matches
                if entities:
                    memory_entity = memory.metadata.get("entity", "").lower()
                    memory_entities = [tag.lower() for tag in memory.tags]
                    
                    for entity in entities:
                        entity_lower = entity.lower()
                        
                        # Direct entity match
                        if entity_lower == memory_entity:
                            score += 0.5
                            match_reason = f"Direct entity match: {entity}"
                        
                        # Tag match
                        elif entity_lower in memory_entities:
                            score += 0.3
                            match_reason = f"Tag match: {entity}"
                        
                        # Partial entity match
                        elif entity_lower in content_lower:
                            score += 0.2
                            match_reason = f"Contains entity: {entity}"
                
                # Boost core memories
                if memory.source == MemoryType.CORE.value:
                    score *= 1.5
                    
                # Boost by importance
                importance = memory.metadata.get("importance", 5)
                score *= (1.0 + (importance / 10.0))
                
                # Only include if somewhat relevant
                if score > 0.1:
                    scored_memories.append((memory, score, match_reason))
            
            # Sort by score and take top results
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = [(memory, score) for memory, score, _ in scored_memories[:limit]]
            
            logger.info(f"Found {len(top_memories)} relevant memories for query")
            return top_memories
            
        except Exception as e:
            logger.error(f"Error finding relevant memories: {str(e)}")
            raise DatabaseError(f"Failed to find relevant memories: {str(e)}")
    
    async def update_memory(
        self,
        memory_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID to update
            user_id: User ID for authorization
            updates: Fields to update
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If memory not found
            AuthorizationError: If user doesn't own the memory
        """
        try:
            # Get current memory
            memory = await self.get_memory(memory_id, user_id)
            
            if not memory:
                raise NotFoundException(f"Memory {memory_id} not found")
            
            # Apply updates
            update_data = {}
            
            if "content" in updates:
                update_data["content"] = updates["content"]
            
            if "metadata" in updates:
                # Merge metadata
                new_metadata = {**memory.metadata, **updates["metadata"]}
                update_data["metadata"] = new_metadata
            
            if "tags" in updates:
                update_data["tags"] = updates["tags"]
            
            # Always update timestamp
            update_data["updated_at"] = datetime.utcnow()
            update_data["is_synced"] = False
            
            # Update in database
            success = await self.db.update_memory(memory_id, update_data)
            
            if success:
                logger.info(f"Updated memory {memory_id} for user {user_id}")
            
            return success
            
        except Exception as e:
            if isinstance(e, (NotFoundException, AuthorizationError)):
                raise
                
            logger.error(f"Error updating memory {memory_id}: {str(e)}")
            raise DatabaseError(f"Failed to update memory: {str(e)}")
    
    async def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID to delete
            user_id: User ID for authorization
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If memory not found
            AuthorizationError: If user doesn't own the memory
        """
        try:
            # Verify ownership first
            memory = await self.get_memory(memory_id, user_id)
            
            if not memory:
                raise NotFoundException(f"Memory {memory_id} not found")
            
            # Delete memory
            success = await self.db.delete_memory(user_id, memory_id)
            
            if success:
                logger.info(f"Deleted memory {memory_id} for user {user_id}")
            
            return success
            
        except Exception as e:
            if isinstance(e, (NotFoundException, AuthorizationError)):
                raise
                
            logger.error(f"Error deleting memory {memory_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete memory: {str(e)}")
    
    async def complete_event(self, memory_id: str, user_id: str) -> bool:
        """
        Mark an event memory as completed.
        
        Args:
            memory_id: Memory ID to update
            user_id: User ID for authorization
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If memory not found
            AuthorizationError: If user doesn't own the memory
        """
        try:
            # Get memory
            memory = await self.get_memory(memory_id, user_id)
            
            if not memory:
                raise NotFoundException(f"Memory {memory_id} not found")
                
            # Verify it's an event memory
            if memory.source != MemoryType.EVENT.value:
                logger.warning(f"Attempted to complete non-event memory {memory_id}")
                return False
            
            # Update metadata
            updates = {
                "metadata": {"completed": True}
            }
            
            return await self.update_memory(memory_id, user_id, updates)
            
        except Exception as e:
            if isinstance(e, (NotFoundException, AuthorizationError)):
                raise
                
            logger.error(f"Error completing event {memory_id}: {str(e)}")
            raise DatabaseError(f"Failed to complete event: {str(e)}")
    
    async def extract_memory_command(self, text: str) -> Optional[MemoryCommand]:
        """
        Extract memory commands from text.
        
        Args:
            text: Text to analyze for commands
            
        Returns:
            Optional[MemoryCommand]: Memory command if found
        """
        # Simple rule-based extraction - in production, use NLP or LLM
        text_lower = text.lower()
        
        # Check for remember command
        if text_lower.startswith("remember "):
            content = text[9:].strip()
            return MemoryCommand(
                command_type="remember",
                content=content,
                # Other fields would be extracted by a more sophisticated system
            )
            
        # Check for remind command
        elif text_lower.startswith("remind me "):
            content = text[10:].strip()
            # Extract time using regex (simplified)
            import re
            time_match = re.search(r'at (\d{1,2}(?::\d{2})? ?(?:am|pm)?)', content, re.IGNORECASE)
            
            event_time = None
            if time_match:
                time_str = time_match.group(1)
                # Simple time parsing (would be more robust in production)
                try:
                    now = datetime.now()
                    if ":" in time_str:
                        hours, mins = time_str.split(":")
                        hours = int(hours)
                        mins = int(mins.replace("am", "").replace("pm", "").strip())
                        
                        if "pm" in time_str.lower() and hours < 12:
                            hours += 12
                            
                        event_time = now.replace(hour=hours, minute=mins)
                    else:
                        hours = int(time_str.replace("am", "").replace("pm", "").strip())
                        
                        if "pm" in time_str.lower() and hours < 12:
                            hours += 12
                            
                        event_time = now.replace(hour=hours, minute=0)
                        
                    # If time is in the past, move to tomorrow
                    if event_time < now:
                        event_time = event_time + timedelta(days=1)
                        
                except Exception as e:
                    logger.warning(f"Failed to parse time from reminder: {str(e)}")
            
            return MemoryCommand(
                command_type="remind",
                content=content,
                event_time=event_time
            )
            
        # Check for forget command
        elif text_lower.startswith("forget "):
            content = text[7:].strip()
            return MemoryCommand(
                command_type="forget",
                content=content
            )
            
        return None


# Singleton instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """
    Get the memory manager singleton.
    
    Returns:
        MemoryManager: Memory manager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
