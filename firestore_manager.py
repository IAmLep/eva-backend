import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
from firebase_admin import firestore
from google.cloud.exceptions import NotFound
from google.api_core.exceptions import PermissionDenied, ServerError

class FirestoreManager:
    """Manager for Firestore database operations with lazy initialization"""
    
    _instance = None
    _db = None
    
    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the manager (but not the client yet)"""
        if not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.logger.info("FirestoreManager initialized (client will be created on first use)")
            self._initialized = True
    
    @property
    def db(self):
        """Lazy-initialized Firestore client property
        
        Only creates the client when actually accessed, allowing Firebase to be
        initialized before the client is created.
        """
        if FirestoreManager._db is None:
            self.logger.info("Creating Firestore client")
            try:
                FirestoreManager._db = firestore.client()
                self.logger.info("Firestore client created successfully")
            except ValueError as e:
                self.logger.error(f"Firebase not initialized: {str(e)}")
                self.logger.error("Make sure firebase_admin.initialize_app() is called before accessing Firestore")
                raise ValueError("Firebase not initialized. Call firebase_admin.initialize_app() first") from e
            except Exception as e:
                self.logger.error(f"Failed to create Firestore client: {str(e)}")
                raise
        return FirestoreManager._db

    # Helper method for validating conversation ownership
    async def verify_conversation_ownership(self, conversation_id: str, user_id: str) -> bool:
        """Verify that a conversation belongs to a user"""
        try:
            conversation_ref = self.db.collection("conversations").document(conversation_id)
            conversation = conversation_ref.get()
            
            if not conversation.exists:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return False
                
            conversation_data = conversation.to_dict()
            return conversation_data.get("user_id") == user_id
        except NotFound:
            self.logger.error(f"Conversation {conversation_id} not found")
            return False
        except PermissionDenied:
            self.logger.error(f"Permission denied when accessing conversation {conversation_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error verifying conversation ownership: {str(e)}")
            return False

    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get a device from Firestore"""
        try:
            device_ref = self.db.collection("devices").document(device_id)
            device = device_ref.get()
            return device.to_dict() if device.exists else None
        except NotFound:
            self.logger.error(f"Device {device_id} not found")
            return None
        except PermissionDenied:
            self.logger.error(f"Permission denied when accessing device {device_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting device {device_id}: {str(e)}")
            return None

    async def add_or_update_device(self, device_id: str, device_data: Dict[str, Any]) -> bool:
        """Add or update a device in Firestore"""
        try:
            device_ref = self.db.collection("devices").document(device_id)
            device_ref.set(device_data, merge=True)
            self.logger.info(f"Added/updated device {device_id}")
            return True
        except PermissionDenied:
            self.logger.error(f"Permission denied when adding/updating device {device_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding/updating device {device_id}: {str(e)}")
            return False

    async def delete_device(self, device_id: str) -> bool:
        """Delete a device from Firestore"""
        try:
            device_ref = self.db.collection("devices").document(device_id)
            device_ref.delete()
            self.logger.info(f"Deleted device {device_id}")
            return True
        except NotFound:
            self.logger.error(f"Device {device_id} not found for deletion")
            return False
        except PermissionDenied:
            self.logger.error(f"Permission denied when deleting device {device_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting device {device_id}: {str(e)}")
            return False

    async def update_device_sync_time(self, device_id: str, sync_time: str) -> bool:
        """Update a device's last sync time"""
        try:
            device_ref = self.db.collection("devices").document(device_id)
            device_ref.update({"last_sync_time": sync_time})
            self.logger.info(f"Updated sync time for device {device_id}")
            return True
        except NotFound:
            self.logger.error(f"Device {device_id} not found for sync time update")
            return False
        except PermissionDenied:
            self.logger.error(f"Permission denied when updating sync time for device {device_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error updating sync time for device {device_id}: {str(e)}")
            return False

    async def add_active_token(self, token_id: str, device_id: str, expiry: int) -> bool:
        """Add an active token to Firestore"""
        try:
            token_ref = self.db.collection("active_tokens").document(token_id)
            token_ref.set({
                "device_id": device_id,
                "expiry": expiry,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            self.logger.info(f"Added active token {token_id} for device {device_id}")
            return True
        except PermissionDenied:
            self.logger.error(f"Permission denied when adding active token {token_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding active token {token_id}: {str(e)}")
            return False

    async def revoke_token(self, token_id: str) -> bool:
        """Add a token to the revoked tokens collection"""
        try:
            revoked_ref = self.db.collection("revoked_tokens").document(token_id)
            revoked_ref.set({
                "revoked_at": firestore.SERVER_TIMESTAMP
            })
            self.logger.info(f"Revoked token {token_id}")
            return True
        except PermissionDenied:
            self.logger.error(f"Permission denied when revoking token {token_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error revoking token {token_id}: {str(e)}")
            return False

    async def is_token_revoked(self, token_id: str) -> bool:
        """Check if a token has been revoked"""
        try:
            revoked_ref = self.db.collection("revoked_tokens").document(token_id)
            revoked = revoked_ref.get()
            is_revoked = revoked.exists
            if is_revoked:
                self.logger.info(f"Token {token_id} is revoked")
            return is_revoked
        except NotFound:
            self.logger.error(f"Token {token_id} not found in revoked tokens")
            return False
        except PermissionDenied:
            self.logger.error(f"Permission denied when checking token {token_id}")
            # Default to treating as revoked if we can't verify
            return True
        except Exception as e:
            self.logger.error(f"Error checking if token {token_id} is revoked: {str(e)}")
            # Default to treating as revoked if there's an error
            return True

    async def add_or_update_conversation(self, conversation_id: str, conversation_data: Dict[str, Any]) -> bool:
        """Add or update a conversation in Firestore"""
        try:
            conversation_ref = self.db.collection("conversations").document(conversation_id)
            conversation_ref.set(conversation_data, merge=True)
            self.logger.info(f"Added/updated conversation {conversation_id}")
            return True
        except PermissionDenied:
            self.logger.error(f"Permission denied when adding/updating conversation {conversation_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding/updating conversation {conversation_id}: {str(e)}")
            return False

    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation from Firestore"""
        try:
            # Verify conversation ownership
            if not await self.verify_conversation_ownership(conversation_id, user_id):
                self.logger.warning(f"User {user_id} does not own conversation {conversation_id}")
                return None
                
            conversation_ref = self.db.collection("conversations").document(conversation_id)
            conversation = conversation_ref.get()
            
            if not conversation.exists:
                self.logger.warning(f"Conversation {conversation_id} not found")
                return None
                
            return conversation.to_dict()
        except NotFound:
            self.logger.error(f"Conversation {conversation_id} not found")
            return None
        except PermissionDenied:
            self.logger.error(f"Permission denied when accessing conversation {conversation_id}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting conversation {conversation_id}: {str(e)}")
            return None

    async def get_conversations_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        try:
            conversations_ref = self.db.collection("conversations").where("user_id", "==", user_id)
            conversations = conversations_ref.stream()
            
            result = []
            for conv in conversations:
                conv_data = conv.to_dict()
                conv_data["id"] = conv.id
                result.append(conv_data)
                
            return result
        except PermissionDenied:
            self.logger.error(f"Permission denied when accessing conversations for user {user_id}")
            return []
        except Exception as e:
            self.logger.error(f"Error getting conversations for user {user_id}: {str(e)}")
            return []

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation from Firestore"""
        try:
            # Verify conversation ownership
            if not await self.verify_conversation_ownership(conversation_id, user_id):
                self.logger.warning(f"User {user_id} does not own conversation {conversation_id}")
                return False
                
            conversation_ref = self.db.collection("conversations").document(conversation_id)
            conversation_ref.delete()
            self.logger.info(f"Deleted conversation {conversation_id}")
            return True
        except NotFound:
            self.logger.error(f"Conversation {conversation_id} not found for deletion")
            return False
        except PermissionDenied:
            self.logger.error(f"Permission denied when deleting conversation {conversation_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error deleting conversation {conversation_id}: {str(e)}")
            return False

    async def add_sync_record(self, source_device_id: str, user_id: str, data_type: str, data: Dict[str, Any]) -> str:
        """Add a sync record to Firestore"""
        try:
            record_id = str(uuid.uuid4())
            sync_ref = self.db.collection("sync_records").document(record_id)
            
            sync_data = {
                "source_device_id": source_device_id,
                "user_id": user_id,
                "data_type": data_type,
                "data": data,
                "synced_devices": [source_device_id],  # Mark as synced for source device
                "created_at": firestore.SERVER_TIMESTAMP
            }
            
            sync_ref.set(sync_data)
            self.logger.info(f"Added sync record {record_id} for user {user_id}")
            return record_id
        except PermissionDenied:
            self.logger.error(f"Permission denied when adding sync record for user {user_id}")
            return ""
        except Exception as e:
            self.logger.error(f"Error adding sync record for user {user_id}: {str(e)}")
            return ""

    async def get_sync_records_for_device(self, device_id: str, user_id: str, last_sync_time: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get sync records for a device since last sync"""
        try:
            # Convert string timestamp to datetime for comparison
            try:
                last_sync_dt = datetime.fromisoformat(last_sync_time)
            except ValueError:
                self.logger.error(f"Invalid last_sync_time format: {last_sync_time}")
                last_sync_dt = datetime.fromisoformat("1970-01-01T00:00:00")
            
            # Query records for this user that don't include this device in synced_devices
            sync_ref = (self.db.collection("sync_records")
                        .where("user_id", "==", user_id)
                        .where("synced_devices", "not-in", [device_id]))
            
            records = sync_ref.stream()
            
            # Group records by data_type
            result = {}
            for record in records:
                record_data = record.to_dict()
                
                # Skip records from this device
                if record_data.get("source_device_id") == device_id:
                    continue
                    
                # Skip records created before last_sync_time
                created_at = record_data.get("created_at")
                if created_at and created_at.timestamp() < last_sync_dt.timestamp():
                    continue
                
                # Add record to results, grouped by data_type
                data_type = record_data.get("data_type", "unknown")
                if data_type not in result:
                    result[data_type] = []
                
                # Add the record ID to the data for tracking
                record_data["data"]["record_id"] = record.id
                result[data_type].append(record_data["data"])
            
            return result
        except PermissionDenied:
            self.logger.error(f"Permission denied when getting sync records for device {device_id}")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting sync records for device {device_id}: {str(e)}")
            return {}

    async def mark_sync_record_as_synced(self, record_id: str, device_id: str) -> bool:
        """Mark a sync record as synced for a specific device"""
        try:
            sync_ref = self.db.collection("sync_records").document(record_id)
            sync_record = sync_ref.get()
            
            if not sync_record.exists:
                self.logger.warning(f"Sync record {record_id} not found")
                return False
                
            # Add this device to the synced_devices array
            sync_ref.update({
                "synced_devices": firestore.ArrayUnion([device_id])
            })
            
            self.logger.info(f"Marked sync record {record_id} as synced for device {device_id}")
            return True
        except NotFound:
            self.logger.error(f"Sync record {record_id} not found")
            return False
        except PermissionDenied:
            self.logger.error(f"Permission denied when marking sync record {record_id}")
            return False
        except Exception as e:
            self.logger.error(f"Error marking sync record {record_id} as synced: {str(e)}")
            return False

# Create a singleton instance
firestore_manager = FirestoreManager()