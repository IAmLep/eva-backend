"""
Database module for EVA backend application.

This module provides database operations for the offline-first approach,
supporting local device storage as primary and Firestore as backup/sync.

Last updated: 2025-04-01
Version: v1.2
"""

import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set, Tuple, Union

from fastapi import HTTPException
from firebase_admin import firestore
from pydantic import BaseModel

from config import get_settings
from exceptions import DatabaseError, NotFoundException
from firestore_manager import get_firestore_client
from models import User, UserInDB, Memory, SyncState

# Logger configuration
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database manager for handling data operations.
    
    This class implements the offline-first approach with local
    data store as primary and Firestore for synchronization.
    """
    
    def __init__(self):
        """Initialize database manager with Firestore client."""
        self.settings = get_settings()
        self.firestore_client = get_firestore_client()
        self.users_collection = "users"
        self.memories_collection = "memories"
        self.sync_states_collection = "sync_states"
        logger.info("Database manager initialized")
    
    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions.
        
        Yields:
            transaction: Transaction object
        
        Raises:
            DatabaseError: If transaction fails
        """
        transaction = self.firestore_client.transaction()
        try:
            yield transaction
            # Transaction automatically committed on context exit
        except Exception as e:
            logger.error(f"Transaction failed: {str(e)}")
            raise DatabaseError(f"Database transaction failed: {str(e)}")
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            Optional[User]: User if found, None otherwise
        """
        try:
            doc_ref = self.firestore_client.collection(self.users_collection).document(user_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.info(f"User not found: {user_id}")
                return None
                
            user_data = doc.to_dict()
            return User(**user_data)
        except Exception as e:
            logger.error(f"Error retrieving user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve user: {str(e)}")
    
    async def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """
        Get user by username.
        
        Args:
            username: Username to retrieve
            
        Returns:
            Optional[UserInDB]: User if found, None otherwise
        """
        try:
            query = (
                self.firestore_client.collection(self.users_collection)
                .where("username", "==", username)
                .limit(1)
            )
            
            docs = query.stream()
            user_doc = next(docs, None)
            
            if not user_doc:
                logger.info(f"User not found by username: {username}")
                return None
                
            user_data = user_doc.to_dict()
            return UserInDB(**user_data)
        except Exception as e:
            logger.error(f"Error retrieving user by username {username}: {str(e)}")
            raise DatabaseError(f"Failed to retrieve user by username: {str(e)}")
    
    async def create_user(self, user: UserInDB) -> str:
        """
        Create a new user.
        
        Args:
            user: User to create
            
        Returns:
            str: ID of created user
            
        Raises:
            DatabaseError: If user creation fails
        """
        try:
            # Check if username already exists
            existing_user = await self.get_user_by_username(user.username)
            if existing_user:
                logger.warning(f"Username already exists: {user.username}")
                raise DatabaseError("Username already exists")
            
            # Add user to Firestore
            user_ref = self.firestore_client.collection(self.users_collection).document()
            user_data = user.model_dump()
            user_ref.set(user_data)
            
            logger.info(f"Created user: {user.username} with ID: {user_ref.id}")
            return user_ref.id
        except DatabaseError:
            # Re-raise database errors
            raise
        except Exception as e:
            logger.error(f"Error creating user {user.username}: {str(e)}")
            raise DatabaseError(f"Failed to create user: {str(e)}")
    
    async def update_user(self, user_id: str, user_data: Dict) -> bool:
        """
        Update user data.
        
        Args:
            user_id: User ID to update
            user_data: Updated user data
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If user not found
            DatabaseError: If update fails
        """
        try:
            # Verify user exists
            user_ref = self.firestore_client.collection(self.users_collection).document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                logger.warning(f"User not found for update: {user_id}")
                raise NotFoundException(f"User {user_id} not found")
            
            # Update user
            user_ref.update(user_data)
            logger.info(f"Updated user: {user_id}")
            return True
        except NotFoundException:
            # Re-raise not found exception
            raise
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to update user: {str(e)}")
    
    async def delete_user(self, user_id: str) -> bool:
        """
        Delete a user.
        
        Args:
            user_id: User ID to delete
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If user not found
            DatabaseError: If deletion fails
        """
        try:
            # Verify user exists
            user_ref = self.firestore_client.collection(self.users_collection).document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                logger.warning(f"User not found for deletion: {user_id}")
                raise NotFoundException(f"User {user_id} not found")
            
            # Delete user
            user_ref.delete()
            logger.info(f"Deleted user: {user_id}")
            return True
        except NotFoundException:
            # Re-raise not found exception
            raise
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete user: {str(e)}")
    
    async def sync_memories(self, user_id: str, memories: List[Memory], 
                           device_id: str) -> Tuple[List[Memory], SyncState]:
        """
        Synchronize memories between local device and Firestore.
        
        Implements the offline-first approach, merging changes and
        tracking sync state to prevent duplicates.
        
        Args:
            user_id: User ID owning the memories
            memories: List of memories from device
            device_id: Device ID for sync tracking
            
        Returns:
            Tuple[List[Memory], SyncState]: 
                - Updated memories list
                - New sync state
                
        Raises:
            DatabaseError: If sync fails
        """
        try:
            # Get current sync state
            sync_ref = (self.firestore_client
                        .collection(self.sync_states_collection)
                        .document(f"{user_id}_{device_id}"))
            sync_doc = sync_ref.get()
            
            current_sync: Optional[SyncState] = None
            if sync_doc.exists:
                current_sync = SyncState(**sync_doc.to_dict())
            else:
                current_sync = SyncState(
                    user_id=user_id,
                    device_id=device_id,
                    last_sync=None,
                    synced_memory_ids=[]
                )
            
            # Get server memories
            query = (self.firestore_client
                     .collection(self.memories_collection)
                     .where("user_id", "==", user_id))
            server_memories = [Memory(**doc.to_dict()) for doc in query.stream()]
            
            # Merge memories (server and device)
            # Skip duplicates based on memory_id in synced_memory_ids
            merged_memories: List[Memory] = []
            synced_ids: Set[str] = set(current_sync.synced_memory_ids)
            
            # Add server memories not in device
            for server_mem in server_memories:
                if server_mem.memory_id not in [m.memory_id for m in memories]:
                    merged_memories.append(server_mem)
                    synced_ids.add(server_mem.memory_id)
            
            # Add device memories not yet synced
            for device_mem in memories:
                if device_mem.memory_id not in synced_ids:
                    # Add to Firestore
                    mem_ref = (self.firestore_client
                              .collection(self.memories_collection)
                              .document(device_mem.memory_id))
                    mem_ref.set(device_mem.model_dump())
                    synced_ids.add(device_mem.memory_id)
                
                # Always include device memories in result
                merged_memories.append(device_mem)
            
            # Update sync state
            new_sync = SyncState(
                user_id=user_id,
                device_id=device_id,
                last_sync=firestore.SERVER_TIMESTAMP,
                synced_memory_ids=list(synced_ids)
            )
            sync_ref.set(new_sync.model_dump())
            
            logger.info(f"Synced {len(memories)} memories for user {user_id}, device {device_id}")
            return merged_memories, new_sync
        except Exception as e:
            logger.error(f"Error syncing memories for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to sync memories: {str(e)}")
    
    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            user_id: User ID owning the memory
            memory_id: Memory ID to delete
            
        Returns:
            bool: True if successful
            
        Raises:
            NotFoundException: If memory not found
            DatabaseError: If deletion fails
        """
        try:
            # Verify memory exists and belongs to user
            mem_ref = self.firestore_client.collection(self.memories_collection).document(memory_id)
            mem_doc = mem_ref.get()
            
            if not mem_doc.exists:
                logger.warning(f"Memory not found for deletion: {memory_id}")
                raise NotFoundException(f"Memory {memory_id} not found")
            
            mem_data = mem_doc.to_dict()
            if mem_data["user_id"] != user_id:
                logger.warning(f"Unauthorized deletion attempt for memory {memory_id} by user {user_id}")
                raise HTTPException(status_code=403, detail="Not authorized to delete this memory")
            
            # Delete memory
            mem_ref.delete()
            
            # Update sync states to remove this memory ID
            sync_query = (self.firestore_client
                         .collection(self.sync_states_collection)
                         .where("user_id", "==", user_id))
            
            for sync_doc in sync_query.stream():
                sync_data = sync_doc.to_dict()
                if memory_id in sync_data["synced_memory_ids"]:
                    synced_ids = set(sync_data["synced_memory_ids"])
                    synced_ids.remove(memory_id)
                    sync_doc.reference.update({"synced_memory_ids": list(synced_ids)})
            
            logger.info(f"Deleted memory {memory_id} for user {user_id}")
            return True
        except NotFoundException:
            # Re-raise not found exception
            raise
        except HTTPException:
            # Re-raise HTTP exception
            raise
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {str(e)}")
            raise DatabaseError(f"Failed to delete memory: {str(e)}")
    
    async def cleanup_old_memories(self, user_id: str, days_threshold: int = 30) -> int:
        """
        Clean up old synced memories to prevent Firestore duplication.
        
        Args:
            user_id: User ID owning the memories
            days_threshold: Age threshold in days for memories to delete
            
        Returns:
            int: Number of memories cleaned up
            
        Raises:
            DatabaseError: If cleanup fails
        """
        try:
            # Calculate cutoff date
            cutoff_date = (
                firestore.firestore.SERVER_TIMESTAMP - 
                firestore.firestore.timedelta(days=days_threshold)
            )
            
            # Find old memories
            query = (self.firestore_client
                    .collection(self.memories_collection)
                    .where("user_id", "==", user_id)
                    .where("created_at", "<", cutoff_date))
            
            # Delete old memories in batch
            batch = self.firestore_client.batch()
            count = 0
            
            for mem_doc in query.stream():
                batch.delete(mem_doc.reference)
                count += 1
                
                # Firestore batches limited to 500 operations
                if count % 400 == 0:
                    batch.commit()
                    batch = self.firestore_client.batch()
            
            # Commit any remaining operations
            if count % 400 != 0:
                batch.commit()
            
            logger.info(f"Cleaned up {count} old memories for user {user_id}")
            return count
        except Exception as e:
            logger.error(f"Error cleaning up old memories for user {user_id}: {str(e)}")
            raise DatabaseError(f"Failed to clean up old memories: {str(e)}")


# Initialize database manager
_db_manager = None


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


# Helper functions for database operations
async def get_user_by_username(username: str) -> Optional[UserInDB]:
    """
    Get user by username.
    
    Args:
        username: Username to retrieve
        
    Returns:
        Optional[UserInDB]: User if found, None otherwise
    """
    db = get_db_manager()
    return await db.get_user_by_username(username)


async def verify_user_exists(user_id: str) -> bool:
    """
    Verify if user exists.
    
    Args:
        user_id: User ID to verify
        
    Returns:
        bool: True if user exists, False otherwise
    """
    db = get_db_manager()
    user = await db.get_user_by_id(user_id)
    return user is not None