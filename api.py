from fastapi import APIRouter, WebSocket, Depends, HTTPException, WebSocketDisconnect, Request, status
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from datetime import datetime
import uuid

from auth import validate_device_token
from firestore_manager import FirestoreManager
from websocket_manager import WebSocketManager
from llm_service import LLMService
from memory_extractor import MemoryExtractor
from cache_manager import CacheManager
from rate_limiter import RateLimiter

router = APIRouter()
firestore = FirestoreManager()
manager = WebSocketManager()
llm_service = LLMService()
memory_extractor = MemoryExtractor()
cache_manager = CacheManager()
rate_limiter = RateLimiter()

logger = logging.getLogger(__name__)

@router.websocket("/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for voice conversations"""
    # Get the token from the query params
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Missing token")
        return
    
    # Connect with token validation
    connection_id = await manager.connect_with_token(websocket, token)
    if not connection_id:
        # Connection failed, websocket already closed
        return
    
    try:
        device_id = manager.device_ids.get(connection_id)
        user_id = manager.user_ids.get(connection_id)
        
        # Start conversation or use existing one
        conversation_id = websocket.query_params.get("conversation_id")
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
            # Create new conversation in Firestore
            await firestore.add_or_update_conversation(
                conversation_id, 
                user_id,
                {
                    "title": "Voice conversation",
                    "created_at": firestore.server_timestamp(),
                    "updated_at": firestore.server_timestamp(),
                    "messages": []
                }
            )
        
        # Send initial connection confirmation
        await manager.send_message(connection_id, json.dumps({
            "type": "connected",
            "conversation_id": conversation_id
        }))
        
        # Rate limiting check
        await rate_limiter.check_rate_limit(user_id or device_id)
        
        # Main message loop
        while True:
            try:
                # Wait for message from user
                data = await websocket.receive_text()
                message_data = json.loads(data)
                
                # Extract message content and type
                content = message_data.get("content", "")
                message_type = message_data.get("type", "text")
                
                # Validate rate limiting for each message
                if not await rate_limiter.record_request(user_id or device_id):
                    await manager.send_message(connection_id, json.dumps({
                        "type": "error",
                        "error": "Rate limit exceeded"
                    }))
                    continue
                
                # Store user message
                message = {
                    "role": "user",
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": message_type
                }
                
                # Retrieve conversation memory
                memory = await memory_extractor.get_conversation_memory(conversation_id, user_id)
                
                # Generate response using Gemini
                # For streaming voice, we might want to send partial results
                response_text = await llm_service.generate_text_streaming(
                    content,
                    memory=memory,
                    callback=lambda chunk: manager.send_message(connection_id, json.dumps({
                        "type": "partial_response",
                        "content": chunk
                    }))
                )
                
                # Store assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "type": "text"  # Assuming text response
                }
                
                # Update conversation in Firestore with both messages
                conversation = await firestore.get_conversation(conversation_id, user_id)
                if conversation:
                    messages = conversation.get("messages", [])
                    messages.append(message)
                    messages.append(assistant_message)
                    
                    # Update title if this is the first message
                    title_update = {}
                    if len(messages) <= 2:
                        title = content[:30] + "..." if len(content) > 30 else content
                        title_update = {"title": title}
                    
                    await firestore.add_or_update_conversation(
                        conversation_id,
                        user_id,
                        {
                            "messages": messages,
                            "updated_at": firestore.server_timestamp(),
                            **title_update
                        }
                    )
                
                # Send complete response
                await manager.send_message(connection_id, json.dumps({
                    "type": "response",
                    "content": response_text,
                    "conversation_id": conversation_id
                }))
                
                # Periodically extract memory (e.g., every 5 messages)
                if len(messages) % 5 == 0:
                    asyncio.create_task(memory_extractor.extract_key_info(conversation_id, user_id))
                
            except json.JSONDecodeError:
                await manager.send_message(connection_id, json.dumps({
                    "type": "error",
                    "error": "Invalid JSON"
                }))
            except Exception as e:
                logger.error(f"Error in voice websocket: {str(e)}")
                await manager.send_message(connection_id, json.dumps({
                    "type": "error",
                    "error": f"Server error: {str(e)}"
                }))
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"Unexpected error in voice websocket: {str(e)}")
        try:
            manager.disconnect(connection_id)
        except:
            pass