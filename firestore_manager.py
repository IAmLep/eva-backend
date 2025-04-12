"""
Firestore Manager for EVA backend.

This module provides Firestore integration for data synchronization and backup
supporting the offline-first approach.


Version 3 working
"""

import logging
import os
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore import Client as FirestoreClient
from google.cloud.firestore_v1.base_query import FieldFilter
from google.cloud.exceptions import NotFound

from config import get_settings

# Logger configuration
logger = logging.getLogger(__name__)


class FirestoreManager:
    """
    Firestore Manager for handling Firestore operations.
    
    This class provides methods for interacting with Firestore collections
    and documents, with specific support for the offline-first sync approach.
    """
    
    def __init__(self, client: Optional[FirestoreClient] = None):
        """
        Initialize Firestore Manager.
        
        Args:
            client: Optional existing Firestore client
        """
        self.settings = get_settings()
        self.client = client or self._initialize_client()
        self._collections = {
            "users": self.client.collection("users"),
            "memories": self.client.collection("memories"),
            "sync_states": self.client.collection("sync_states"),
            "devices": self.client.collection("devices"),
        }
        logger.info("Firestore Manager initialized")
    
    def _initialize_client(self) -> FirestoreClient:
        """
        Initialize Firestore client.
        
        Returns:
            FirestoreClient: Initialized Firestore client
            
        Raises:
            RuntimeError: If initialization fails
        """
        settings = self.settings
        
        try:
            # Check if Firebase app is already initialized
            firebase_admin.get_app()
        except ValueError:
            # Initialize Firebase app
            if settings.is_production:
                # In production, use default credentials
                # This assumes application is running in Google Cloud environment
                # with appropriate IAM permissions
                firebase_admin.initialize_app()
                logger.info("Firebase initialized with default credentials")
            else:
                # For development or testing, use a service account or emulator
                if settings.FIRESTORE_EMULATOR_HOST:
                    # Use emulator
                    os.environ["FIRESTORE_EMULATOR_HOST"] = settings.FIRESTORE_EMULATOR_HOST
                    firebase_admin.initialize_app()
                    logger.info(f"Firebase initialized with emulator at {settings.FIRESTORE_EMULATOR_HOST}")
                else:
                    # Use service account if available
                    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    if cred_path:
                        cred = credentials.Certificate(cred_path)
                        firebase_admin.initialize_app(cred, {
                            'projectId': settings.GOOGLE_CLOUD_PROJECT,
                        })
                        logger.info(f"Firebase initialized with service account from {cred_path}")
                    else:
                        firebase_admin.initialize_app()
                        logger.warning("Firebase initialized with default credentials in non-production environment")
        
        # Get Firestore client
        return firestore.client()
    
    def collection(self, name: str):
        """
        Get a Firestore collection reference.
        
        Args:
            name: Collection name
            
        Returns:
            Collection reference
        """
        if name not in self._collections:
            self._collections[name] = self.client.collection(name)
        return self._collections[name]
    
    async def get_document(
        self, 
        collection_name: str, 
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document from Firestore.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            
        Returns:
            Optional[Dict[str, Any]]: Document data or None if not found
        """
        try:
            doc_ref = self.collection(collection_name).document(document_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                logger.info(f"Document not found: {collection_name}/{document_id}")
                return None
                
            return doc.to_dict()
        except Exception as e:
            logger.error(f"Error getting document {collection_name}/{document_id}: {str(e)}")
            return None
    
    async def set_document(
        self, 
        collection_name: str, 
        document_id: str, 
        data: Dict[str, Any], 
        merge: bool = False
    ) -> bool:
        """
        Set a document in Firestore.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            data: Document data
            merge: Whether to merge with existing document
            
        Returns:
            bool: Success status
        """
        try:
            doc_ref = self.collection(collection_name).document(document_id)
            doc_ref.set(data, merge=merge)
            logger.info(f"Document set: {collection_name}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting document {collection_name}/{document_id}: {str(e)}")
            return False
    
    async def update_document(
        self, 
        collection_name: str, 
        document_id: str, 
        data: Dict[str, Any]
    ) -> bool:
        """
        Update a document in Firestore.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            data: Document data
            
        Returns:
            bool: Success status
        """
        try:
            doc_ref = self.collection(collection_name).document(document_id)
            doc_ref.update(data)
            logger.info(f"Document updated: {collection_name}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {collection_name}/{document_id}: {str(e)}")
            return False
    
    async def delete_document(
        self, 
        collection_name: str, 
        document_id: str
    ) -> bool:
        """
        Delete a document from Firestore.
        
        Args:
            collection_name: Collection name
            document_id: Document ID
            
        Returns:
            bool: Success status
        """
        try:
            doc_ref = self.collection(collection_name).document(document_id)
            doc_ref.delete()
            logger.info(f"Document deleted: {collection_name}/{document_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {collection_name}/{document_id}: {str(e)}")
            return False
    
    async def query_documents(
        self, 
        collection_name: str, 
        filters: List[Tuple[str, str, Any]], 
        limit: Optional[int] = None,
        order_by: Optional[List[Tuple[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query documents from Firestore.
        
        Args:
            collection_name: Collection name
            filters: List of filter tuples (field, operator, value)
            limit: Optional result limit
            order_by: Optional ordering [(field, direction)]
            
        Returns:
            List[Dict[str, Any]]: List of document data
        """
        try:
            collection_ref = self.collection(collection_name)
            query = collection_ref
            
            # Apply filters
            for field, op, value in filters:
                query = query.where(filter=FieldFilter(field, op, value))
            
            # Apply ordering
            if order_by:
                for field, direction in order_by:
                    if direction.lower() == "desc":
                        query = query.order_by(field, direction=firestore.Query.DESCENDING)
                    else:
                        query = query.order_by(field)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Execute query
            results = query.stream()
            documents = [doc.to_dict() for doc in results]
            
            logger.info(f"Query returned {len(documents)} documents from {collection_name}")
            return documents
        except Exception as e:
            logger.error(f"Error querying documents from {collection_name}: {str(e)}")
            return []
    
    async def batch_write(
        self, 
        operations: List[Dict[str, Any]]
    ) -> bool:
        """
        Perform batch write operations.
        
        Args:
            operations: List of operations with format:
                {
                    "type": "set"|"update"|"delete",
                    "collection": collection_name,
                    "document_id": document_id,
                    "data": data_dict  # for set/update only
                }
            
        Returns:
            bool: Success status
        """
        try:
            batch = self.client.batch()
            
            for op in operations:
                op_type = op.get("type")
                collection = op.get("collection")
                doc_id = op.get("document_id")
                
                if not all([op_type, collection, doc_id]):
                    logger.warning(f"Invalid batch operation: {op}")
                    continue
                
                doc_ref = self.collection(collection).document(doc_id)
                
                if op_type == "set":
                    data = op.get("data", {})
                    merge = op.get("merge", False)
                    batch.set(doc_ref, data, merge=merge)
                elif op_type == "update":
                    data = op.get("data", {})
                    batch.update(doc_ref, data)
                elif op_type == "delete":
                    batch.delete(doc_ref)
                else:
                    logger.warning(f"Unknown operation type: {op_type}")
            
            # Commit batch
            batch.commit()
            logger.info(f"Batch write completed with {len(operations)} operations")
            return True
        except Exception as e:
            logger.error(f"Error in batch write: {str(e)}")
            return False
    
    async def sync_memories(
        self,
        user_id: str,
        device_id: str,
        memories: List[Dict[str, Any]],
        last_sync: Optional[datetime] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Synchronize memories between device and Firestore.
        
        Implements the offline-first approach for memory synchronization.
        
        Args:
            user_id: User ID
            device_id: Device ID
            memories: List of memories from device
            last_sync: Optional timestamp of last sync
            
        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, Any]]:
                - Updated memories list
                - Sync state
        """
        try:
            # Get current sync state
            sync_id = f"{user_id}_{device_id}"
            sync_data = await self.get_document("sync_states", sync_id)
            
            if not sync_data:
                # Initialize sync state
                sync_data = {
                    "user_id": user_id,
                    "device_id": device_id,
                    "last_sync": None,
                    "synced_memory_ids": []
                }
            
            # Get server memories updated since last sync
            server_memories = []
            if last_sync:
                server_memories = await self.query_documents(
                    "memories",
                    [
                        ("user_id", "==", user_id),
                        ("updated_at", ">=", last_sync)
                    ]
                )
            else:
                # If no last_sync, get all user's memories
                server_memories = await self.query_documents(
                    "memories",
                    [("user_id", "==", user_id)]
                )
            
            # Merge memories (server and device)
            synced_ids = set(sync_data.get("synced_memory_ids", []))
            merged_memories = []
            batch_operations = []
            
            # Memory IDs from device
            device_memory_ids = {mem["memory_id"] for mem in memories}
            
            # Add server memories not in device
            for server_mem in server_memories:
                if server_mem["memory_id"] not in device_memory_ids:
                    merged_memories.append(server_mem)
                    synced_ids.add(server_mem["memory_id"])
            
            # Add device memories not yet synced or updated
            for device_mem in memories:
                memory_id = device_mem["memory_id"]
                
                # Find if memory exists on server
                server_mem = next(
                    (mem for mem in server_memories if mem["memory_id"] == memory_id), 
                    None
                )
                
                if not server_mem:
                    # Memory doesn't exist on server, add it
                    batch_operations.append({
                        "type": "set",
                        "collection": "memories",
                        "document_id": memory_id,
                        "data": device_mem
                    })
                    synced_ids.add(memory_id)
                elif device_mem.get("updated_at") > server_mem.get("updated_at"):
                    # Device memory is newer, update server
                    batch_operations.append({
                        "type": "update",
                        "collection": "memories",
                        "document_id": memory_id,
                        "data": device_mem
                    })
                    synced_ids.add(memory_id)
                
                # Always include device memories in result
                merged_memories.append(device_mem)
            
            # Add memories from server that were deleted on device
            deleted_memory_ids = synced_ids - device_memory_ids
            for memory_id in deleted_memory_ids:
                batch_operations.append({
                    "type": "delete",
                    "collection": "memories",
                    "document_id": memory_id
                })
            
            # Remove deleted memories from synced IDs
            synced_ids -= deleted_memory_ids
            
            # Update sync state
            now = datetime.utcnow()
            sync_data.update({
                "last_sync": now,
                "synced_memory_ids": list(synced_ids)
            })
            
            batch_operations.append({
                "type": "set",
                "collection": "sync_states",
                "document_id": sync_id,
                "data": sync_data,
                "merge": True
            })
            
            # Execute batch operations
            if batch_operations:
                await self.batch_write(batch_operations)
            
            logger.info(f"Synced {len(memories)} memories for user {user_id}, device {device_id}")
            return merged_memories, sync_data
        except Exception as e:
            logger.error(f"Error syncing memories: {str(e)}")
            raise
    
    async def cleanup_old_synced_memories(
        self,
        user_id: str,
        days_threshold: int = 30
    ) -> int:
        """
        Clean up old synced memories to prevent Firestore duplication.
        
        Args:
            user_id: User ID
            days_threshold: Age threshold in days
            
        Returns:
            int: Number of memories cleaned up
        """
        try:
            # Calculate cutoff date
            cutoff_date = datetime.utcnow().replace(
                day=datetime.utcnow().day - days_threshold
            )
            
            # Get memories older than threshold
            old_memories = await self.query_documents(
                "memories",
                [
                    ("user_id", "==", user_id),
                    ("created_at", "<", cutoff_date),
                    ("synced", "==", True)  # Only delete memories that are synced
                ]
            )
            
            if not old_memories:
                logger.info(f"No old memories to clean up for user {user_id}")
                return 0
            
            # Delete old memories in batches
            batch_size = 500
            for i in range(0, len(old_memories), batch_size):
                batch = old_memories[i:i+batch_size]
                batch_operations = [
                    {
                        "type": "delete",
                        "collection": "memories",
                        "document_id": mem["memory_id"]
                    }
                    for mem in batch
                ]
                
                await self.batch_write(batch_operations)
            
            # Update sync states to remove these memory IDs
            sync_docs = await self.query_documents(
                "sync_states",
                [("user_id", "==", user_id)]
            )
            
            old_memory_ids = {mem["memory_id"] for mem in old_memories}
            
            for sync_doc in sync_docs:
                synced_ids = set(sync_doc.get("synced_memory_ids", []))
                removed_ids = synced_ids.intersection(old_memory_ids)
                
                if removed_ids:
                    new_synced_ids = list(synced_ids - removed_ids)
                    await self.update_document(
                        "sync_states",
                        f"{user_id}_{sync_doc['device_id']}",
                        {"synced_memory_ids": new_synced_ids}
                    )
            
            logger.info(f"Cleaned up {len(old_memories)} old memories for user {user_id}")
            return len(old_memories)
        except Exception as e:
            logger.error(f"Error cleaning up old memories: {str(e)}")
            return 0
    
    async def register_device(
        self,
        user_id: str,
        device_id: str,
        device_info: Dict[str, Any]
    ) -> bool:
        """
        Register a device for a user.
        
        Args:
            user_id: User ID
            device_id: Device ID
            device_info: Device information
            
        Returns:
            bool: Success status
        """
        try:
            device_data = {
                "user_id": user_id,
                "device_id": device_id,
                "last_active": datetime.utcnow(),
                **device_info
            }
            
            success = await self.set_document(
                "devices",
                f"{user_id}_{device_id}",
                device_data,
                merge=True
            )
            
            logger.info(f"Device {device_id} registered for user {user_id}")
            return success
        except Exception as e:
            logger.error(f"Error registering device: {str(e)}")
            return False


# Singleton instance
_firestore_manager = None


@lru_cache()
def get_firestore_client() -> FirestoreClient:
    """
    Get Firestore client.
    
    Returns:
        FirestoreClient: Firestore client instance
    """
    manager = get_firestore_manager()
    return manager.client


def get_firestore_manager() -> FirestoreManager:
    """
    Get Firestore manager singleton.
    
    Returns:
        FirestoreManager: Firestore manager instance
    """
    global _firestore_manager
    if _firestore_manager is None:
        _firestore_manager = FirestoreManager()
    return _firestore_manager