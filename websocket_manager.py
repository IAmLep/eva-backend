import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Callable
from fastapi import WebSocket, WebSocketDisconnect, status
from datetime import datetime
from config import WS_HEARTBEAT_INTERVAL
from auth import validate_device_token
from firestore_manager import store_document, get_document, update_document, query_collection

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        # Device ID -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Conversation ID -> List of device IDs
        self.conversation_listeners: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, device_id: str):
        await websocket.accept()
        self.active_connections[device_id] = websocket
        
        # Store connection info in Firestore
        try:
            await store_document("websocket_connections", device_id, {
                "connected_at": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat(),
                "active": True,
                "user_agent": websocket.headers.get("user-agent", "unknown")
            })
        except Exception as e:
            logger.error(f"Failed to store connection in Firestore: {e}")
        
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
        try:
            await store_document("websocket_connections", device_id, {
                "connected_at": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat(),
                "active": True,
                "user_agent": websocket.headers.get("user-agent", "unknown"),
                "token_validated": True
            })
        except Exception as e:
            logger.error(f"Failed to store connection in Firestore: {e}")
            
        return True
        
    async def disconnect(self, device_id: str):
        if device_id in self.active_connections:
            del self.active_connections[device_id]
            
        # Update connection status in Firestore
        try:
            await update_document("websocket_connections", device_id, {
                "disconnected_at": datetime.utcnow().isoformat(),
                "active": False
            })
        except Exception as e:
            logger.error(f"Failed to update disconnection in Firestore: {e}")
            
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
                    logger.error(f"Failed to update conversation participation in Firestore: {e}")
    
    async def join_conversation(self, conversation_id: str, device_id: str):
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
            except Exception as e:
                logger.error(f"Failed to store conversation participation in Firestore: {e}")
    
    async def leave_conversation(self, conversation_id: str, device_id: str):
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
            except Exception as e:
                logger.error(f"Failed to update conversation participation in Firestore: {e}")
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Any):
        if conversation_id not in self.conversation_listeners:
            return
            
        # Store message in Firestore if it's a new message
        try:
            if isinstance(message, dict) and message.get("type") == "new_message":
                msg = message.get("message", {})
                await store_document(
                    "conversation_messages", 
                    f"{conversation_id}_{datetime.utcnow().isoformat()}",
                    {
                        "conversation_id": conversation_id,
                        "role": msg.get("role", "unknown"),
                        "content": msg.get("content", ""),
                        "device_id": msg.get("device_id", "system"),
                        "timestamp": msg.get("created_at", datetime.utcnow().isoformat())
                    }
                )
        except Exception as e:
            logger.error(f"Failed to store message in Firestore: {e}")
            
        # Send to all connected devices
        for device_id in self.conversation_listeners[conversation_id]:
            if device_id in self.active_connections:
                try:
                    await self.active_connections[device_id].send_json(message)
                except Exception as e:
                    logger.error(f"Failed to send message to device {device_id}: {e}")
                    await self.disconnect(device_id)
    
    async def send_to_device(self, device_id: str, message: Any):
        if device_id in self.active_connections:
            await self.active_connections[device_id].send_json(message)
            
    async def broadcast_heartbeat(self):
        """Send heartbeat to all connected clients periodically"""
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
                        logger.error(f"Failed to update heartbeat in Firestore: {e}")
                        
                except Exception as e:
                    # Connection probably closed
                    logger.warning(f"Failed to send heartbeat to {device_id}: {e}")
                    await self.disconnect(device_id)
            
            await asyncio.sleep(WS_HEARTBEAT_INTERVAL)
    
    async def restore_from_firestore(self):
        """Restore conversation participation from Firestore on startup"""
        try:
            participants = await query_collection(
                "conversation_participants",
                [("active", "==", True)]
            )
            
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
            logger.error(f"Failed to restore state from Firestore: {e}")

# Create a singleton instance
manager = ConnectionManager()