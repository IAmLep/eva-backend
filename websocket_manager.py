import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect, status
from datetime import datetime
from config import WS_HEARTBEAT_INTERVAL
from auth import validate_device_token
from firestore_manager import store_document, get_document, update_document, query_collection, delete_document

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Device ID -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Conversation ID -> List of device IDs
        self.conversation_listeners: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, device_id: str):
        """
        Connect a WebSocket client
        """
        await websocket.accept()
        self.active_connections[device_id] = websocket
        
        # Store connection info in Firestore
        await store_document("websocket_connections", device_id, {
            "connected_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "active": True,
            "user_agent": websocket.headers.get("user-agent", "unknown")
        })
        logger.debug(f"Device {device_id} connected and stored in Firestore")
        
    async def connect_with_token(self, websocket: WebSocket, device_id: str, token: str) -> bool:
        """
        Connect a WebSocket with token validation
        
        Args:
            websocket: The WebSocket connection
            device_id: Device ID attempting to connect
            token: Authentication token
            
        Returns:
            True if connection was successful, False otherwise
        """
        # Validate the token
        validation = validate_device_token(token)
        
        if not validation.valid:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid token")
            return False
            
        # Verify device ID matches token subject
        if validation.device_id != device_id:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Device ID mismatch")
            return False
            
        # Accept connection
        await websocket.accept()
        self.active_connections[device_id] = websocket
        
        # Store connection info in Firestore
        await store_document("websocket_connections", device_id, {
            "connected_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "active": True,
            "user_agent": websocket.headers.get("user-agent", "unknown"),
            "token_validated": True
        })
        
        logger.debug(f"Device {device_id} connected with token validation and stored in Firestore")
        return True
        
    async def disconnect(self, device_id: str):
        """
        Disconnect a client and update Firestore
        """
        if device_id in self.active_connections:
            del self.active_connections[device_id]
            
        # Mark as disconnected in Firestore
        try:
            await update_document("websocket_connections", device_id, {
                "disconnected_at": datetime.utcnow().isoformat(),
                "active": False
            })
            logger.debug(f"Device {device_id} disconnected and updated in Firestore")
        except Exception as e:
            logger.error(f"Error updating disconnect status in Firestore: {e}")
            
        # Remove device from all conversation listeners
        for conv_id in self.conversation_listeners:
            if device_id in self.conversation_listeners[conv_id]:
                self.conversation_listeners[conv_id].remove(device_id)
                
                # Update conversation participation in Firestore
                try:
                    doc_id = f"{conv_id}_{device_id}"
                    await update_document("conversation_participants", doc_id, {
                        "left_at": datetime.utcnow().isoformat(),
                        "active": False
                    })
                except Exception as e:
                    logger.error(f"Error updating conversation participation in Firestore: {e}")
    
    async def join_conversation(self, conversation_id: str, device_id: str):
        """
        Add a device to a conversation with Firestore persistence
        """
        if conversation_id not in self.conversation_listeners:
            self.conversation_listeners[conversation_id] = []
            
        if device_id not in self.conversation_listeners[conversation_id]:
            self.conversation_listeners[conversation_id].append(device_id)
            
            # Store in Firestore
            try:
                doc_id = f"{conversation_id}_{device_id}"
                await store_document("conversation_participants", doc_id, {
                    "conversation_id": conversation_id,
                    "device_id": device_id,
                    "joined_at": datetime.utcnow().isoformat(),
                    "active": True
                })
                logger.debug(f"Device {device_id} joined conversation {conversation_id} and stored in Firestore")
            except Exception as e:
                logger.error(f"Error storing conversation participation in Firestore: {e}")
    
    async def leave_conversation(self, conversation_id: str, device_id: str):
        """
        Remove a device from a conversation with Firestore update
        """
        if (conversation_id in self.conversation_listeners and 
            device_id in self.conversation_listeners[conversation_id]):
            self.conversation_listeners[conversation_id].remove(device_id)
            
            # Update in Firestore
            try:
                doc_id = f"{conversation_id}_{device_id}"
                await update_document("conversation_participants", doc_id, {
                    "left_at": datetime.utcnow().isoformat(),
                    "active": False
                })
                logger.debug(f"Device {device_id} left conversation {conversation_id} and updated in Firestore")
            except Exception as e:
                logger.error(f"Error updating conversation participation in Firestore: {e}")
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Any):
        """
        Broadcast a message to all devices in a conversation
        """
        if conversation_id not in self.conversation_listeners:
            return
        
        # Store message in conversation history if it's a user or assistant message
        if isinstance(message, dict) and message.get("type") == "new_message":
            try:
                message_data = message.get("message", {})
                await store_document(
                    "conversation_messages", 
                    f"{conversation_id}_{datetime.utcnow().isoformat()}",
                    {
                        "conversation_id": conversation_id,
                        "role": message_data.get("role", "unknown"),
                        "content": message_data.get("content", ""),
                        "device_id": message_data.get("device_id", "system"),
                        "timestamp": message_data.get("created_at", datetime.utcnow().isoformat())
                    }
                )
            except Exception as e:
                logger.error(f"Error storing message in Firestore: {e}")
            
        # Send to all active connections
        for device_id in self.conversation_listeners[conversation_id]:
            if device_id in self.active_connections:
                await self.active_connections[device_id].send_json(message)
    
    async def send_to_device(self, device_id: str, message: Any):
        """
        Send a message to a specific device
        """
        if device_id in self.active_connections:
            await self.active_connections[device_id].send_json(message)
            
    async def broadcast_heartbeat(self):
        """
        Send heartbeat to all connected clients periodically and update Firestore
        """
        while True:
            timestamp = datetime.utcnow().isoformat()
            for device_id, connection in list(self.active_connections.items()):
                try:
                    await connection.send_json({"type": "heartbeat", "timestamp": timestamp})
                    
                    # Update heartbeat in Firestore
                    try:
                        await update_document("websocket_connections", device_id, {
                            "last_heartbeat": timestamp,
                            "active": True
                        })
                    except Exception as e:
                        logger.error(f"Error updating heartbeat in Firestore: {e}")
                        
                except Exception as e:
                    # Connection probably closed
                    logger.warning(f"Heartbeat failed for device {device_id}: {e}")
                    await self.disconnect(device_id)
            
            await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
            
    async def restore_from_firestore(self):
        """
        Restore conversation participation information from Firestore
        This is useful when restarting the service to maintain state about who was in which conversation
        """
        try:
            # Find all active conversation participants
            participants = await query_collection(
                "conversation_participants",
                [("active", "==", True)]
            )
            
            # Rebuild conversation listeners map
            for participant in participants:
                conversation_id = participant.get("conversation_id")
                device_id = participant.get("device_id")
                
                if conversation_id and device_id:
                    if conversation_id not in self.conversation_listeners:
                        self.conversation_listeners[conversation_id] = []
                    
                    if device_id not in self.conversation_listeners[conversation_id]:
                        self.conversation_listeners[conversation_id].append(device_id)
                        
            logger.info(f"Restored {len(participants)} conversation participants from Firestore")
            
        except Exception as e:
            logger.error(f"Error restoring from Firestore: {e}")

# Create a singleton instance
manager = ConnectionManager()