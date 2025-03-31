import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from typing import Dict, Any, List, Optional
import os
import logging
from datetime import datetime

class FirestoreManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FirestoreManager, cls).__new__(cls)
            
            # Initialize only once
            try:
                # Check if already initialized
                firebase_admin.get_app()
            except ValueError:
                # Initialize with credentials
                cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                if cred_path and os.path.exists(cred_path):
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                else:
                    # Use application default credentials
                    firebase_admin.initialize_app()
            
            cls._instance.db = firestore.client()
            cls._instance.logger = logging.getLogger(__name__)
            
        return cls._instance
    
    def server_timestamp(self):
        """Return a server timestamp object"""
        return firestore.SERVER_TIMESTAMP
    
    # Device methods
    async def add_or_update_device(self, device_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a device in Firestore"""
        device_ref = self.db.collection("devices").document(device_id)
        device_ref.set(data, merge=True)
        return data
    
    async def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get a device from Firestore"""
        device_ref = self.db.collection("devices").document(device_id)
        device = device_ref.get()
        return device.to_dict() if device.exists else None
    
    async def delete_device(self, device_id: str) -> bool:
        """Delete a device from Firestore"""
        self.db.collection("devices").document(device_id).delete()
        return True
    
    # Token management
    async def add_active_token(self, device_id: str, jti: str, expiry: datetime) -> None:
        """Add an active token to Firestore"""
        token_ref = self.db.collection("tokens").document(jti)
        token_ref.set({
            "device_id": device_id,
            "expires_at": expiry,
            "revoked": False,
            "created_at": self.server_timestamp()
        })
    
    async def revoke_token(self, jti: str, device_id: str) -> None:
        """Revoke a token in Firestore"""
        token_ref = self.db.collection("tokens").document(jti)
        token_ref.set({"revoked": True}, merge=True)
    
    async def is_token_revoked(self, jti: str) -> bool:
        """Check if a token is revoked"""
        token_ref = self.db.collection("tokens").document(jti)
        token = token_ref.get()
        if not token.exists:
            return True
        return token.to_dict().get("revoked", False)
    
    # Conversation methods
    async def add_or_update_conversation(self, conversation_id: str, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a conversation in Firestore"""
        # Include user_id in the conversation data
        data["user_id"] = user_id
        
        conversation_ref = self.db.collection("conversations").document(conversation_id)
        conversation_ref.set(data, merge=True)
        return data
    
    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation from Firestore"""
        conversation_ref = self.db.collection("conversations").document(conversation_id)
        conversation = conversation_ref.get()
        
        if not conversation.exists:
            return None
            
        data = conversation.to_dict()
        
        # Verify ownership
        if data.get("user_id") != user_id:
            self.logger.warning(f"User {user_id} attempted to access conversation {conversation_id} owned by {data.get('user_id')}")
            return None
            
        return data
    
    async def get_conversations_by_user(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversations by user ID"""
        conversations = []
        
        query = (
            self.db.collection("conversations")
            .where(filter=FieldFilter("user_id", "==", user_id))
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        
        results = query.stream()
        
        for doc in results:
            conversation = doc.to_dict()
            conversation["id"] = doc.id
            conversations.append(conversation)
            
        return conversations
    
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation from Firestore (soft delete)"""
        # Get the conversation to verify ownership
        conversation = await self.get_conversation(conversation_id, user_id)
        
        if not conversation:
            return False
            
        # Soft delete by updating the status
        conversation_ref = self.db.collection("conversations").document(conversation_id)
        conversation_ref.set({
            "deleted": True,
            "updated_at": self.server_timestamp()
        }, merge=True)
        
        return True

    # Sync methods
    async def add_sync_record(self, record_data: Dict[str, Any], user_id: str) -> str:
        """Add a sync record to Firestore"""
        # Add user_id to the record
        record_data["user_id"] = user_id
        record_data["timestamp"] = self.server_timestamp()
        
        record_ref = self.db.collection("sync_records").document()
        record_ref.set(record_data)
        
        return record_ref.id
    
    async def get_sync_records_for_device(self, device_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Get sync records for a device that are not created by the device"""
        records = []
        
        query = (
            self.db.collection("sync_records")
            .where(filter=FieldFilter("user_id", "==", user_id))
            .where(filter=FieldFilter("origin_device_id", "!=", device_id))
            .order_by("timestamp")
        )
        
        results = query.stream()
        
        for doc in results:
            record = doc.to_dict()
            record["id"] = doc.id
            records.append(record)
            
        return records