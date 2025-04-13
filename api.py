"""
Main API endpoints for EVA backend.

This file has been updated to include:
- Conversation handling with real-time context and memory integration.
Replace your existing `api.py` file with this version.
"""
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Any

from conversation_handler import ConversationHandler
from models import User
from auth import get_current_user

app = FastAPI()

# WebSocket connections
active_connections = {}

class Message(BaseModel):
    content: str

@app.websocket("/ws/conversation")
async def conversation_websocket(websocket: WebSocket, user: User = Depends(get_current_user)):
    """
    WebSocket endpoint for real-time conversation streaming.
    """
    await websocket.accept()
    handler = ConversationHandler(user)
    active_connections[user.user_id] = websocket

    try:
        while True:
            # Receive message from user
            data = await websocket.receive_json()
            message = Message(**data).content

            # Process message
            response = await handler.process_message(message)
            
            # Send response back
            await websocket.send_json({"response": response})
    except WebSocketDisconnect:
        # Handle disconnection
        del active_connections[user.user_id]
    except Exception as e:
        await websocket.send_json({"error": str(e)})

@app.post("/conversation")
async def post_conversation(message: Message, user: User = Depends(get_current_user)):
    """
    REST endpoint for conversation interactions.
    """
    handler = ConversationHandler(user)
    response = await handler.process_message(message.content)
    return {"response": response}