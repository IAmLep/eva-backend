import asyncio
import json
import logging
from typing import Dict, Set, Any, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
import jwt
from datetime import datetime

from auth import validate_device_token
from firestore_manager import FirestoreManager

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.client_info: Dict[WebSocket, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        self.firestore = FirestoreManager()

    async def connect_with_token(self, websocket: WebSocket, token: str) -> Dict[str, Any]:
        """Connect a client with JWT authentication"""
        try:
            # Validate the token and get device info
            device_info = await validate_device_token(token)
            
            if not device_info:
                self.logger.warning("Invalid token during WebSocket connection")
                await websocket.close(code=1008, reason="Invalid authentication token")
                return {}
            
            device_id = device_info.get("device_id")
            user_id = device_info.get("user_id")
            
            if not device_id:
                self.logger.warning("No device_id in token during WebSocket connection")
                await websocket.close(code=1008, reason="Invalid device identification")
                return {}
                
            # Check if device is active
            device = await self.firestore.get_device(device_id)
            if not device or not device.get("is_active", False):
                self.logger.warning(f"Inactive device {device_id} attempted WebSocket connection")
                await websocket.close(code=1008, reason="Device is inactive")
                return {}
            
            # Accept the connection
            await websocket.accept()
            
            # Store the device_id and user_id with the connection
            if device_id not in self.active_connections:
                self.active_connections[device_id] = set()
            self.active_connections[device_id].add(websocket)
            
            self.client_info[websocket] = {
                "device_id": device_id,
                "user_id": user_id,
                "connected_at": datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"WebSocket connected for device {device_id}, user {user_id}")
            return device_info
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Expired token during WebSocket connection")
            await websocket.close(code=1008, reason="Authentication token expired")
            return {}
        except jwt.InvalidTokenError:
            self.logger.warning("Invalid token format during WebSocket connection") 
            await websocket.close(code=1008, reason="Invalid authentication token")
            return {}
        except Exception as e:
            self.logger.error(f"Error in connect_with_token: {str(e)}")
            await websocket.close(code=1011, reason="Server error during authentication")
            return {}

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a client"""
        try:
            if websocket in self.client_info:
                device_id = self.client_info[websocket].get("device_id")
                user_id = self.client_info[websocket].get("user_id")
                
                if device_id and device_id in self.active_connections:
                    self.active_connections[device_id].remove(websocket)
                    if len(self.active_connections[device_id]) == 0:
                        del self.active_connections[device_id]
                
                del self.client_info[websocket]
                self.logger.info(f"WebSocket disconnected for device {device_id}, user {user_id}")
        except Exception as e:
            self.logger.error(f"Error in disconnect: {str(e)}")

    async def send_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client"""
        try:
            await websocket.send_text(message)
        except WebSocketDisconnect:
            self.logger.warning("WebSocket disconnected while sending message")
            await self.disconnect(websocket)
        except RuntimeError as e:
            self.logger.error(f"Runtime error sending message: {str(e)}")
            await self.disconnect(websocket)
        except Exception as e:
            self.logger.error(f"Error in send_message: {str(e)}")
            await self.disconnect(websocket)

    async def broadcast(self, message: str, device_id: str):
        """Send a message to all clients for a specific device"""
        if device_id not in self.active_connections:
            return
            
        disconnected_websockets = []
        
        for websocket in self.active_connections[device_id]:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                self.logger.warning(f"WebSocket disconnected during broadcast to device {device_id}")
                disconnected_websockets.append(websocket)
            except RuntimeError as e:
                self.logger.error(f"Runtime error during broadcast: {str(e)}")
                disconnected_websockets.append(websocket)
            except Exception as e:
                self.logger.error(f"Error in broadcast: {str(e)}")
                disconnected_websockets.append(websocket)
        
        # Cleanup disconnected websockets
        for ws in disconnected_websockets:
            await self.disconnect(ws)
            
    async def broadcast_to_user(self, message: str, user_id: str):
        """Send a message to all clients for a specific user"""
        user_websockets = []
        
        # Find all websockets for this user
        for websocket, info in self.client_info.items():
            if info.get("user_id") == user_id:
                user_websockets.append(websocket)
        
        disconnected_websockets = []
        
        for websocket in user_websockets:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                self.logger.warning(f"WebSocket disconnected during broadcast to user {user_id}")
                disconnected_websockets.append(websocket)
            except RuntimeError as e:
                self.logger.error(f"Runtime error during user broadcast: {str(e)}")
                disconnected_websockets.append(websocket)
            except Exception as e:
                self.logger.error(f"Error in user broadcast: {str(e)}")
                disconnected_websockets.append(websocket)
        
        # Cleanup disconnected websockets
        for ws in disconnected_websockets:
            await self.disconnect(ws)

    async def restore_from_firestore(self):
        """Restore needed data from Firestore on startup"""
        try:
            # Any initialization needed from Firestore
            self.logger.info("Restored WebSocketManager from Firestore")
        except Exception as e:
            self.logger.error(f"Error restoring from Firestore: {str(e)}")