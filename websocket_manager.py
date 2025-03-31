import asyncio
import logging
from typing import Dict, Set, Callable, Awaitable, Any, Optional
import json
from fastapi import WebSocket
from auth import validate_device_token

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manager for WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.device_ids: Dict[str, str] = {}  # Maps connection_id to device_id
        self.user_ids: Dict[str, str] = {}  # Maps connection_id to user_id
        self.connection_ids: Dict[str, str] = {}  # Maps device_id to connection_id
    
    async def connect_with_token(self, websocket: WebSocket, token: str) -> Optional[str]:
        """
        Connect a WebSocket with token validation
        Returns connection_id if successful, None if validation fails
        """
        try:
            # Validate the token
            payload = await validate_device_token(token)
            if not payload:
                await websocket.close(code=1008, reason="Invalid token")
                return None
            
            device_id = payload.get("sub")
            user_id = payload.get("user_id")  # If you're storing user_id in tokens
            
            if not device_id:
                await websocket.close(code=1008, reason="Invalid device ID")
                return None
            
            # Accept the connection
            await websocket.accept()
            
            # Generate a unique connection ID
            import uuid
            connection_id = str(uuid.uuid4())
            
            # Store the connection
            self.active_connections[connection_id] = websocket
            self.device_ids[connection_id] = device_id
            
            if user_id:
                self.user_ids[connection_id] = user_id
                
            # Map device_id to connection_id for easy lookup
            self.connection_ids[device_id] = connection_id
            
            logger.info(f"WebSocket connected: {connection_id} (Device: {device_id}, User: {user_id})")
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {str(e)}")
            try:
                await websocket.close(code=1011, reason="Server error")
            except:
                pass
            return None
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket by connection ID"""
        if connection_id in self.active_connections:
            device_id = self.device_ids.get(connection_id)
            user_id = self.user_ids.get(connection_id)
            
            # Remove from all mappings
            if device_id:
                self.connection_ids.pop(device_id, None)
            
            self.active_connections.pop(connection_id, None)
            self.device_ids.pop(connection_id, None)
            self.user_ids.pop(connection_id, None)
            
            logger.info(f"WebSocket disconnected: {connection_id} (Device: {device_id}, User: {user_id})")
    
    async def send_message(self, connection_id: str, message: str):
        """Send a message to a WebSocket by connection ID"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all active connections"""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {str(e)}")
                disconnected.append(connection_id)
        
        # Clean up disconnected websockets
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    async def send_to_device(self, device_id: str, message: str) -> bool:
        """Send a message to a specific device"""
        connection_id = self.connection_ids.get(device_id)
        if not connection_id:
            return False
            
        await self.send_message(connection_id, message)
        return True