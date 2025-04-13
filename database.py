"""
Database Manager for EVA backend.

This module provides database operations for storing and retrieving
data, with enhanced support for the multi-tiered memory system.

Update your existing database.py file with this version.

Current Date: 2025-04-13 11:03:01
Current User: IAmLepin
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore as firebase_firestore
from pydantic import BaseModel

from config import get_settings
from models import Memory, User, Conversation, SyncState, ApiKey, UserRateLimit

# Logger configuration
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for Firestore.
    
    This class provides methods for database operations with
    enhanced support for the multi-tiered memory system.
    """
    
    def __init__(self):
        """Initialize database manager with Firestore client."""
        settings = get_settings()
        self.is_production = settings.ENVIRONMENT == "production"
        
        try:
            # Initialize Firestore client
            if not firebase_admin._apps:
                # Use service account credentials if available
                credentials_path = settings.FIREBASE_CREDENTIALS_PATH
                if os.path.exists(credentials_path):
                    cred = credentials.Certificate(credentials_path)
                    firebase_admin.initialize_app(cred)
                else:
                    # Use application default credentials
                    firebase_admin.initialize_app()
            
            # Get Firestore client
            self.db = firebase_firestore.client()
            logger.info("Firestore database initialized")
        
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            # Set up a dummy db for development if needed
            self.db = None
            self.in_memory_db = {
                "users": {},
                "memories": {},
                "conversations": {},
                "sync_states": {},
                "api_keys": {},
                "rate_limits": {}
            }
            logger.warning("Using in-memory database (for development only)")
    
    async def create_user(self, user: User) -> bool:
        """
        Create a new user.
        
        Args:
            user: User object to create
            
        Returns:
            bool: Success status
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            if self.db:
                # Convert to dict and save
                user_data = user.model_dump(exclude={"id"})
                self.db.collection("users").document(user.id).set(user_data)
            else:
                # In-memory storage
                self.in_memory_db["users"][user.id] = user.model_dump()
            
            logger.info(f"Created user {user.id}: {user.username}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            from exceptions import DatabaseError
            raise DatabaseError(f"Failed to create user: {str(e)}")
    
    async def get_user(self, user_id: str) -> Optional[User]:
        """
        Get a user by ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        try:
            if self.db:
                # Get from Firestore
                doc = self.db.collection("users").document(user_id).get()
                if doc.exists:
                    user_data = doc.to_dict()
                    # Add ID from document
                    user_data["id"] = doc.id
                    return User(**user_data)
                else:
                    return None
            else:
                # Get from in-memory storage
                user_data = self.in_memory_db["users"].get(user_id)
                return User(**user_data) if user_data else None
            
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {str(e)}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get a user by email.
        
        Args:
            email: User email to look up
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        try:
            if self.db:
                # Query Firestore
                query = self.db.collection("users").where("email", "==", email).limit(1)
                results = query.get()
                
                for doc in results:
                    user_data = doc.to_dict()
                    # Add ID from document
                    user_data["id"] = doc.id
                    return User(**user_data)
                
                return None
            else:
                # Query in-memory storage
                for user_id, user_data in self.in_memory_db["users"].items():
                    if user_data.get("email") == email:
                        return User(**user_data)
                
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving user by email {email}: {str(e)}")
            return None
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a user.
        
        Args:
            user_id: User ID to update
            updates: Fields to update
            
        Returns:
            bool: Success status
        """
        try:
            # Always update timestamp
            updates["updated_at"] = datetime.utcnow()
            
            if self.db:
                # Update in Firestore
                self.db.collection("users").document(user_id).update(updates)
            else:
                # Update in-memory storage
                if user_id in self.in_memory_db["users"]:
                    self.in_memory_db["users"][user_id].update(updates)
                else:
                    return False
            
            logger.info(f"Updated user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {str(e)}")
            return False
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID to delete
            
        Returns:
            bool: Success status
        """
        try:
            if self.db:
                # Delete from Firestore
                self.db.collection("users").document(user_id).delete()
            else:
                # Delete from in-memory storage
                if user_id in self.in_memory_db["users"]:
                    del self.in_memory_db["users"][user_id]
                else:
                    return False
            
            logger.info(f"Deleted user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {str(e)}")
            return False
    
    # Memory Operations
    
    async def create_memory(self, memory: Memory) -> bool:
        """
        Create a new memory.
        
        Args:
            memory: Memory object to create
            
        Returns:
            bool: Success status
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            # Convert to dict for storage
            memory_data = memory.model_dump()
            
            if self.db:
                # Store in Firestore
                self.db.collection("memories").document(memory.memory_id).set(memory_data)
            else:
                # Store in-memory
                self.in_memory_db["memories"][memory.memory_id] = memory_data
            
            logger.info(f"Created memory {memory.memory_id} for user {memory.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating memory: {str(e)}")
            from exceptions import DatabaseError
            raise DatabaseError(f"Failed to create memory: {str(e)}")
    
    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Optional[Memory]: Memory if found, None otherwise
        """
        try:
            if self.db:
                # Get from Firestore
                doc = self.db.collection("memories").document(memory_id).get()
                if doc.exists:
                    memory_data = doc.to_dict()
                    return Memory(**memory_data)
                else:
                    return None
            else:
                # Get from in-memory storage
                memory_data = self.in_memory_db["memories"].get(memory_id)
                return Memory(**memory_data) if memory_data else None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {str(e)}")
            return None
    
    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Memory]:
        """
        Get memories for a user.
        
        Args:
            user_id: User ID to get memories for
            limit: Maximum number of memories to return
            offset: Offset for pagination
            memory_type: Optional memory type filter
            tags: Optional tags to filter by
            
        Returns:
            List[Memory]: List of memories
        """
        try:
            memories = []
            
            if self.db:
                # Start with base query
                query = self.db.collection("memories").where("user_id", "==", user_id)
                
                # Add type filter if provided
                if memory_type:
                    query = query.where("source", "==", memory_type)
                
                # TODO: For proper pagination, we'd need to use a cursor
                # This is a simplified approach
                results = query.limit(limit + offset).get()
                
                # Apply offset and limit
                for i, doc in enumerate(results):
                    if i >= offset:  # Skip items before offset
                        memory_data = doc.to_dict()
                        memory = Memory(**memory_data)
                        
                        # Apply tag filtering
                        if tags:
                            # Memory must have all requested tags
                            if all(tag in memory.tags for tag in tags):
                                memories.append(memory)
                        else:
                            memories.append(memory)
                        
                        # Apply limit
                        if len(memories) >= limit:
                            break
            else:
                # Filter in-memory storage
                filtered_memories = []
                
                for memory_data in self.in_memory_db["memories"].values():
                    if memory_data.get("user_id") == user_id:
                        # Apply type filter
                        if memory_type and memory_data.get("source") != memory_type:
                            continue
                            
                        # Apply tag filter
                        if tags:
                            memory_tags = memory_data.get("tags", [])
                            if not all(tag in memory_tags for tag in tags):
                                continue
                        
                        filtered_memories.append(Memory(**memory_data))
                
                # Apply pagination
                memories = filtered_memories[offset:offset + limit]
            
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories for user {user_id}: {str(e)}")
            return []
    
    async def update_memory(
        self,
        memory_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID to update
            updates: Fields to update
            
        Returns:
            bool: Success status
        """
        try:
            if self.db:
                # Update in Firestore
                self.db.collection("memories").document(memory_id).update(updates)
            else:
                # Update in-memory storage
                if memory_id in self.in_memory_db["memories"]:
                    self.in_memory_db["memories"][memory_id].update(updates)
                else:
                    return False
            
            logger.info(f"Updated memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {str(e)}")
            return False
    
    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            user_id: User ID for authorization
            memory_id: Memory ID to delete
            
        Returns:
            bool: Success status
            
        Raises:
            AuthorizationError: If user doesn't own the memory
        """
        try:
            # First verify ownership
            memory = await self.get_memory(memory_id)
            
            if not memory:
                logger.warning(f"Memory {memory_id} not found for deletion")
                return False
                
            if memory.user_id != user_id:
                logger.warning(f"User {user_id} attempted to delete memory {memory_id} owned by {memory.user_id}")
                from exceptions import AuthorizationError
                raise AuthorizationError("Not authorized to delete this memory")
            
            if self.db:
                # Delete from Firestore
                self.db.collection("memories").document(memory_id).delete()
            else:
                # Delete from in-memory storage
                if memory_id in self.in_memory_db["memories"]:
                    del self.in_memory_db["memories"][memory_id]
                else:
                    return False
            
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True
            
        except Exception as e:
            if "AuthorizationError" in str(type(e)):
                raise
                
            logger.error(f"Error deleting memory {memory_id}: {str(e)}")
            return False
    
    async def get_memories_since(
        self, 
        user_id: str, 
        since: datetime
    ) -> List[Memory]:
        """
        Get memories updated since a specific time.
        
        Args:
            user_id: User ID to get memories for
            since: Datetime to get memories since
            
        Returns:
            List[Memory]: List of memories
        """
        try:
            memories = []
            
            if self.db:
                # Query Firestore
                query = (self.db.collection("memories")
                         .where("user_id", "==", user_id)
                         .where("updated_at", ">=", since))
                
                results = query.get()
                
                for doc in results:
                    memory_data = doc.to_dict()
                    memories.append(Memory(**memory_data))
            else:
                # Query in-memory storage
                for memory_data in self.in_memory_db["memories"].values():
                    if memory_data.get("user_id") == user_id:
                        updated_at = memory_data.get("updated_at")
                        if isinstance(updated_at, str):
                            updated_at = datetime.fromisoformat(updated_at)
                            
                        if updated_at >= since:
                            memories.append(Memory(**memory_data))
            
            logger.info(f"Retrieved {len(memories)} memories for user {user_id} since {since}")
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving memories since {since}: {str(e)}")
            return []
    
    async def get_events_by_timerange(
        self,
        user_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[Memory]:
        """
        Get event memories within a time range.
        
        Args:
            user_id: User ID to get events for
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List[Memory]: List of event memories
        """
        try:
            events = []
            
            if self.db:
                # Query Firestore
                query = (self.db.collection("memories")
                         .where("user_id", "==", user_id)
                         .where("source", "==", "event"))
                
                results = query.get()
                
                for doc in results:
                    memory_data = doc.to_dict()
                    
                    # Check if event is within time range
                    event_time_str = memory_data.get("metadata", {}).get("event_time")
                    if event_time_str:
                        try:
                            event_time = datetime.fromisoformat(event_time_str)
                            if start_time <= event_time <= end_time:
                                events.append(Memory(**memory_data))
                        except (ValueError, TypeError):
                            pass
            else:
                # Query in-memory storage
                for memory_data in self.in_memory_db["memories"].values():
                    if (memory_data.get("user_id") == user_id and 
                        memory_data.get("source") == "event"):
                        
                        # Check if event is within time range
                        event_time_str = memory_data.get("metadata", {}).get("event_time")
                        if event_time_str:
                            try:
                                event_time = datetime.fromisoformat(event_time_str)
                                if start_time <= event_time <= end_time:
                                    events.append(Memory(**memory_data))
                            except (ValueError, TypeError):
                                pass
            
            logger.info(f"Retrieved {len(events)} events for user {user_id} between {start_time} and {end_time}")
            return events
            
        except Exception as e:
            logger.error(f"Error retrieving events in time range: {str(e)}")
            return []
    
    # Conversation Operations
    
    async def create_conversation(self, conversation: Conversation) -> bool:
        """
        Create a new conversation.
        
        Args:
            conversation: Conversation object to create
            
        Returns:
            bool: Success status
            
        Raises:
            DatabaseError: If creation fails
        """
        try:
            # Convert to dict for storage
            conversation_data = conversation.model_dump()
            
            if self.db:
                # Store in Firestore
                self.db.collection("conversations").document(conversation.conversation_id).set(conversation_data)
            else:
                # Store in-memory
                self.in_memory_db["conversations"][conversation.conversation_id] = conversation_data
            
            logger.info(f"Created conversation {conversation.conversation_id} for user {conversation.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
            from exceptions import DatabaseError
            raise DatabaseError(f"Failed to create conversation: {str(e)}")
    
    async def get_conversation(
        self, 
        conversation_id: str
    ) -> Optional[Conversation]:
        """
        Get a conversation by ID.
        
        Args:
            conversation_id: Conversation ID to retrieve
            
        Returns:
            Optional[Conversation]: Conversation if found, None otherwise
        """
        try:
            if self.db:
                # Get from Firestore
                doc = self.db.collection("conversations").document(conversation_id).get()
                if doc.exists:
                    conversation_data = doc.to_dict()
                    return Conversation(**conversation_data)
                else:
                    return None
            else:
                # Get from in-memory storage
                conversation_data = self.in_memory_db["conversations"].get(conversation_id)
                return Conversation(**conversation_data) if conversation_data else None
            
        except Exception as e:
            logger.error(f"Error retrieving conversation {conversation_id}: {str(e)}")
            return None
    
    async def get_user_conversations(
        self, 
        user_id: str,
        limit: int = 10,
        offset: int = 0
    ) -> List[Conversation]:
        """
        Get conversations for a user.
        
        Args:
            user_id: User ID to get conversations for
            limit: Maximum number of conversations to return
            offset: Offset for pagination
            
        Returns:
            List[Conversation]: List of conversations
        """
        try:
            conversations = []
            
            if self.db:
                # Query Firestore
                query = (self.db.collection("conversations")
                         .where("user_id", "==", user_id)
                         .order_by("updated_at", direction="DESCENDING")
                         .limit(limit)
                         .offset(offset))
                
                results = query.get()
                
                for doc in results:
                    conversation_data = doc.to_dict()
                    conversations.append(Conversation(**conversation_data))
            else:
                # Filter and sort in-memory storage
                filtered_conversations = []
                
                for conversation_data in self.in_memory_db["conversations"].values():
                    if conversation_data.get("user_id") == user_id:
                        filtered_conversations.append(Conversation(**conversation_data))
                
                # Sort by updated_at
                filtered_conversations.sort(
                    key=lambda c: c.updated_at,
                    reverse=True
                )
                
                # Apply pagination
                conversations = filtered_conversations[offset:offset + limit]
            
            logger.info(f"Retrieved {len(conversations)} conversations for user {user_id}")
            return conversations
            
        except Exception as e:
            logger.error(f"Error retrieving conversations for user {user_id}: {str(e)}")
            return []
    
    async def update_conversation(
        self,
        conversation_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a conversation.
        
        Args:
            conversation_id: Conversation ID to update
            updates: Fields to update
            
        Returns:
            bool: Success status
        """
        try:
            if self.db:
                # Update in Firestore
                self.db.collection("conversations").document(conversation_id).update(updates)
            else:
                # Update in-memory storage
                if conversation_id in self.in_memory_db["conversations"]:
                    self.in_memory_db["conversations"][conversation_id].update(updates)
                else:
                    return False
            
            logger.info(f"Updated conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating conversation {conversation_id}: {str(e)}")
            return False
    
    async def add_message_to_conversation(
        self,
        conversation_id: str,
        message: Dict[str, Any]
    ) -> bool:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID to update
            message: Message to add
            
        Returns:
            bool: Success status
        """
        try:
            if self.db:
                # Use array union to add message
                conversation_ref = self.db.collection("conversations").document(conversation_id)
                
                # Update the conversation
                conversation_ref.update({
                    "messages": firebase_firestore.ArrayUnion([message]),
                    "updated_at": datetime.utcnow()
                })
            else:
                # Update in-memory storage
                if conversation_id in self.in_memory_db["conversations"]:
                    # Add message to array
                    if "messages" not in self.in_memory_db["conversations"][conversation_id]:
                        self.in_memory_db["conversations"][conversation_id]["messages"] = []
                        
                    self.in_memory_db["conversations"][conversation_id]["messages"].append(message)
                    self.in_memory_db["conversations"][conversation_id]["updated_at"] = datetime.utcnow().isoformat()
                else:
                    return False
            
            logger.info(f"Added message to conversation {conversation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {str(e)}")
            return False
    
    async def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        """
        Delete a conversation.
        
        Args:
            user_id: User ID for authorization
            conversation_id: Conversation ID to delete
            
        Returns:
            bool: Success status
            
        Raises:
            AuthorizationError: If user doesn't own the conversation
        """
        try:
            # First verify ownership
            conversation = await self.get_conversation(conversation_id)
            
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found for deletion")
                return False
                
            if conversation.user_id != user_id:
                logger.warning(f"User {user_id} attempted to delete conversation {conversation_id} owned by {conversation.user_id}")
                from exceptions import AuthorizationError
                raise AuthorizationError("Not authorized to delete this conversation")
            
            if self.db:
                # Delete from Firestore
                self.db.collection("conversations").document(conversation_id).delete()
            else:
                # Delete from in-memory storage
                if conversation_id in self.in_memory_db["conversations"]:
                    del self.in_memory_db["conversations"][conversation_id]
                else:
                    return False
            
            logger.info(f"Deleted conversation {conversation_id} for user {user_id}")
            return True
            
        except Exception as e:
            if "AuthorizationError" in str(type(e)):
                raise
                
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            return False
    
    # Sync Operations
    
    async def get_sync_state(
        self, 
        user_id: str, 
        device_id: str
    ) -> Optional[SyncState]:
        """
        Get synchronization state for a device.
        
        Args:
            user_id: User ID
            device_id: Device ID
            
        Returns:
            Optional[SyncState]: Sync state if found, None otherwise
        """
        try:
            if self.db:
                # Get from Firestore
                doc_id = f"{user_id}_{device_id}"
                doc = self.db.collection("sync_states").document(doc_id).get()
                if doc.exists:
                    sync_data = doc.to_dict()
                    return SyncState(**sync_data)
                else:
                    return None
            else:
                # Get from in-memory storage
                doc_id = f"{user_id}_{device_id}"
                sync_data = self.in_memory_db["sync_states"].get(doc_id)
                return SyncState(**sync_data) if sync_data else None
            
        except Exception as e:
            logger.error(f"Error retrieving sync state for {user_id}/{device_id}: {str(e)}")
            return None
    
    async def update_sync_state(self, sync_state: SyncState) -> bool:
        """
        Update synchronization state.
        
        Args:
            sync_state: Sync state to update
            
        Returns:
            bool: Success status
        """
        try:
            # Convert to dict for storage
            sync_data = sync_state.model_dump()
            doc_id = f"{sync_state.user_id}_{sync_state.device_id}"
            
            if self.db:
                # Update in Firestore
                self.db.collection("sync_states").document(doc_id).set(sync_data)
            else:
                # Update in-memory storage
                self.in_memory_db["sync_states"][doc_id] = sync_data
            
            logger.info(f"Updated sync state for {sync_state.user_id}/{sync_state.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating sync state: {str(e)}")
            return False


# Singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the database manager singleton.
    
    Returns:
        DatabaseManager: Database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager