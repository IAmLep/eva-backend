import asyncio
import json
from typing import Dict, List, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
from config import WS_HEARTBEAT_INTERVAL

class ConnectionManager:
    def __init__(self):
        # Device ID -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Conversation ID -> List of device IDs
        self.conversation_listeners: Dict[str, List[str]] = {}
        
    async def connect(self, websocket: WebSocket, device_id: str):
        await websocket.accept()
        self.active_connections[device_id] = websocket
        
    def disconnect(self, device_id: str):
        if device_id in self.active_connections:
            del self.active_connections[device_id]
            
        # Remove device from all conversation listeners
        for conv_id in self.conversation_listeners:
            if device_id in self.conversation_listeners[conv_id]:
                self.conversation_listeners[conv_id].remove(device_id)
    
    def join_conversation(self, conversation_id: str, device_id: str):
        if conversation_id not in self.conversation_listeners:
            self.conversation_listeners[conversation_id] = []
        if device_id not in self.conversation_listeners[conversation_id]:
            self.conversation_listeners[conversation_id].append(device_id)
    
    def leave_conversation(self, conversation_id: str, device_id: str):
        if (conversation_id in self.conversation_listeners and 
            device_id in self.conversation_listeners[conversation_id]):
            self.conversation_listeners[conversation_id].remove(device_id)
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Any):
        if conversation_id not in self.conversation_listeners:
            return
            
        for device_id in self.conversation_listeners[conversation_id]:
            if device_id in self.active_connections:
                await self.active_connections[device_id].send_json(message)
    
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
                except Exception:
                    # Connection probably closed
                    self.disconnect(device_id)
            
            await asyncio.sleep(WS_HEARTBEAT_INTERVAL)

manager = ConnectionManager()