from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import json
import asyncio
import logging
from datetime import datetime
import uuid

from auth import validate_device_token, get_device_from_token
from websocket_manager import ConnectionManager
from llm_service import llm_service
from memory_extractor import memory_extractor
from firestore_manager import firestore_manager
from rate_limiter import RateLimiter
from logging_config import setup_logging
from error_middleware import ErrorMiddleware

# Setup logging
logger = logging.getLogger(__name__)

# Initialize the connection manager for WebSockets
manager = ConnectionManager()

# Initialize rate limiter - customizable limits
rate_limiter = RateLimiter(
    requests_per_minute=20,  # Default for API endpoints
    requests_per_day=1000    # Default daily limit
)

# Voice WebSocket rate limiter - separate limits
voice_rate_limiter = RateLimiter(
    requests_per_minute=60,  # Higher for real-time
    requests_per_day=2000    # Higher daily limit for voice
)

# Models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None
    message_id: Optional[str] = None

class CreateConversationRequest(BaseModel):
    messages: List[ChatMessage]
    conversation_id: Optional[str] = None
    title: Optional[str] = None

class UpdateConversationRequest(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None

class DeleteConversationRequest(BaseModel):
    conversation_id: str

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[ChatMessage]

class ConversationListResponse(BaseModel):
    conversations: List[Dict[str, Any]]

# Initialize the FastAPI application
app = FastAPI(title="EVA API", version="1.8.3")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware
app.add_middleware(ErrorMiddleware)

# Background task for extracting memory
async def extract_memory_task(conversation_id: str, user_id: str):
    """Background task to extract memory from a conversation"""
    try:
        logger.info(f"Starting memory extraction for conversation {conversation_id}")
        await memory_extractor.extract_key_info(conversation_id, user_id)
        logger.info(f"Memory extraction completed for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"Error in memory extraction task: {str(e)}")

# API endpoints
@app.post("/chat", response_model=Dict[str, Any])
async def chat_endpoint(
    request: CreateConversationRequest,
    background_tasks: BackgroundTasks,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Main chat endpoint that processes messages and returns AI responses"""
    # Rate limiting by user ID (fall back to device ID if no user)
    user_id = token_data.get("user_id")
    rate_limit_key = user_id if user_id else token_data["device_id"]
    
    try:
        await rate_limiter.check_rate_limit(rate_limit_key)
        
        # Get or create a conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            # Generate a new conversation ID
            conversation_id = str(uuid.uuid4())
            
            # Default title based on first message
            title = request.title
            if not title and request.messages:
                first_message = request.messages[0].content
                title = first_message[:30] + "..." if len(first_message) > 30 else first_message
            else:
                title = f"Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
            
            # Create the conversation
            new_conversation = {
                "id": conversation_id,
                "title": title,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "messages": [],
                "user_id": user_id,
                "device_id": token_data["device_id"]
            }
            
            await firestore_manager.add_or_update_conversation(conversation_id, new_conversation)
            
            logger.info(f"Created new conversation {conversation_id} for user {user_id}")
        
        # Verify conversation ownership if it already exists
        if conversation_id:
            conversation = await firestore_manager.get_conversation(conversation_id)
            if not conversation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Conversation not found"
                )
            
            # Check ownership
            if conversation.get("user_id") != user_id:
                logger.warning(f"User {user_id} attempted to access conversation {conversation_id} owned by {conversation.get('user_id')}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this conversation"
                )
        
        # Process the messages
        messages = request.messages
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No messages provided"
            )
        
        # Prepare conversation history
        chat_history = []
        if conversation_id and conversation:
            chat_history = conversation.get("messages", [])
        
        # Add new messages to history
        for msg in messages:
            # Generate message ID if not provided
            if not msg.message_id:
                msg.message_id = str(uuid.uuid4())
            
            # Add timestamp if not provided
            if not msg.timestamp:
                msg.timestamp = datetime.utcnow().isoformat()
            
            # Add to chat history
            chat_history.append(msg.dict())
        
        # Get the most recent user message
        user_message = None
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user message found"
            )
        
        # Get conversation memory
        memory_entries = await memory_extractor.get_conversation_memory(conversation_id, user_id)
        memory_context = ""
        if memory_entries:
            # Format memory for the prompt
            memory_points = []
            for entry in memory_entries:
                if "memory_data" in entry:
                    data = entry["memory_data"]
                    if "key_points" in data and data["key_points"]:
                        memory_points.extend(data["key_points"])
                    if "user_preferences" in data and data["user_preferences"]:
                        memory_points.extend([f"Preference: {pref}" for pref in data["user_preferences"]])
            
            if memory_points:
                memory_context = "Previous conversation context:\n" + "\n".join([f"- {point}" for point in memory_points])
        
        # Prepare the prompt with memory context
        prompt = user_message
        if memory_context:
            prompt = f"{memory_context}\n\nUser: {user_message}"
        
        # Generate AI response
        ai_response = await llm_service.generate_text(prompt, chat_history[-10:] if len(chat_history) > 10 else chat_history)
        
        # Create AI message
        ai_message = ChatMessage(
            role="assistant",
            content=ai_response,
            message_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Add AI message to history
        chat_history.append(ai_message.dict())
        
        # Update conversation in Firestore
        conversation_data = {
            "id": conversation_id,
            "title": request.title if request.title else (conversation.get("title") if conversation else title),
            "updated_at": datetime.utcnow().isoformat(),
            "messages": chat_history,
            "user_id": user_id,
            "device_id": token_data["device_id"]
        }
        
        await firestore_manager.add_or_update_conversation(conversation_id, conversation_data)
        
        # Schedule memory extraction if needed
        if len(chat_history) % 5 == 0:  # Extract memory every 5 messages
            background_tasks.add_task(extract_memory_task, conversation_id, user_id)
        
        # Return the AI response and conversation details
        return {
            "message": ai_message.dict(),
            "conversation_id": conversation_id
        }
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {rate_limit_key}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing error: {str(e)}"
        )

@app.websocket("/voice")
async def voice_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time voice conversations"""
    # Accept the connection
    await websocket.accept()
    
    try:
        # Get token from first message
        data = await websocket.receive_text()
        data_json = json.loads(data)
        token = data_json.get("token")
        
        if not token:
            await websocket.send_json({"error": "Authentication required"})
            await websocket.close(code=1008)  # Policy violation
            return
        
        # Validate token
        try:
            token_data = await validate_device_token(token)
            device_id = token_data["device_id"]
            user_id = token_data.get("user_id")
            
            # If no user_id is associated, reject connection
            if not user_id:
                await websocket.send_json({"error": "Device not associated with a user"})
                await websocket.close(code=1008)
                return
            
            # Rate limit check for connection
            rate_limit_key = user_id  # Use user_id for rate limiting
            await voice_rate_limiter.check_rate_limit(rate_limit_key)
            
            # Connect with token validation
            await manager.connect_with_token(websocket, token)
            
            # Send welcome message
            await websocket.send_json({
                "type": "welcome",
                "message": "Connected to voice API"
            })
            
            # Create or get conversation
            conversation_id = data_json.get("conversation_id")
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                conversation = {
                    "id": conversation_id,
                    "title": f"Voice Conversation {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "messages": [],
                    "user_id": user_id,
                    "device_id": device_id,
                    "type": "voice"
                }
                await firestore_manager.add_or_update_conversation(conversation_id, conversation)
                await websocket.send_json({
                    "type": "conversation_created",
                    "conversation_id": conversation_id
                })
            else:
                # Verify conversation ownership
                conversation = await firestore_manager.get_conversation(conversation_id)
                if not conversation or conversation.get("user_id") != user_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Not authorized to access this conversation"
                    })
                    await websocket.close(code=1003)
                    return
            
            # Initialize chat history
            chat_history = conversation.get("messages", [])
            
            # Process messages from the client
            while True:
                # Rate limit check per message
                await voice_rate_limiter.check_rate_limit(rate_limit_key, increment=0.2)  # Lower increment for stream
                
                # Receive message
                data = await websocket.receive_text()
                data_json = json.loads(data)
                message_type = data_json.get("type", "")
                
                # Handle different message types
                if message_type == "user_message":
                    user_message = data_json.get("message", "")
                    if not user_message:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Empty message"
                        })
                        continue
                    
                    # Create user message
                    user_msg = {
                        "role": "user",
                        "content": user_message,
                        "message_id": data_json.get("message_id", str(uuid.uuid4())),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Add to chat history
                    chat_history.append(user_msg)
                    
                    # Get conversation memory
                    memory_entries = await memory_extractor.get_conversation_memory(conversation_id, user_id)
                    memory_context = ""
                    if memory_entries:
                        memory_points = []
                        for entry in memory_entries:
                            if "memory_data" in entry:
                                data = entry["memory_data"]
                                if "key_points" in data and data["key_points"]:
                                    memory_points.extend(data["key_points"])
                                if "user_preferences" in data and data["user_preferences"]:
                                    memory_points.extend([f"Preference: {pref}" for pref in data["user_preferences"]])
                        
                        if memory_points:
                            memory_context = "Previous conversation context:\n" + "\n".join([f"- {point}" for point in memory_points])
                    
                    # Prepare the prompt with memory context
                    prompt = user_message
                    if memory_context:
                        prompt = f"{memory_context}\n\nUser: {user_message}"
                    
                    # Send "thinking" message
                    await websocket.send_json({
                        "type": "thinking",
                        "message": "Generating response..."
                    })
                    
                    # Create AI message ID
                    ai_message_id = str(uuid.uuid4())
                    
                    # Generate streaming response
                    async def stream_callback(chunk: str):
                        try:
                            await websocket.send_json({
                                "type": "assistant_message_chunk",
                                "chunk": chunk,
                                "message_id": ai_message_id
                            })
                        except Exception as e:
                            logger.error(f"Error in stream callback: {str(e)}")
                    
                    # Generate streaming response
                    full_response = ""
                    async for chunk in llm_service.generate_text_streaming(
                        prompt,
                        stream_callback,
                        chat_history[-10:] if len(chat_history) > 10 else chat_history
                    ):
                        full_response += chunk
                    
                    # Create AI message
                    ai_msg = {
                        "role": "assistant",
                        "content": full_response,
                        "message_id": ai_message_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Add to chat history
                    chat_history.append(ai_msg)
                    
                    # Send complete message
                    await websocket.send_json({
                        "type": "assistant_message_complete",
                        "message": full_response,
                        "message_id": ai_message_id
                    })
                    
                    # Update conversation in Firestore
                    conversation["messages"] = chat_history
                    conversation["updated_at"] = datetime.utcnow().isoformat()
                    await firestore_manager.add_or_update_conversation(conversation_id, conversation)
                    
                    # Schedule memory extraction if needed
                    if len(chat_history) % 5 == 0:  # Extract memory every 5 messages
                        asyncio.create_task(extract_memory_task(conversation_id, user_id))
                
                elif message_type == "heartbeat":
                    # Respond to heartbeat
                    await websocket.send_json({
                        "type": "heartbeat_ack"
                    })
                
                elif message_type == "end":
                    # End the conversation
                    await websocket.send_json({
                        "type": "goodbye",
                        "message": "Voice conversation ended"
                    })
                    break
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    })
                
        except HTTPException as e:
            # Handle authentication errors
            await websocket.send_json({
                "type": "error",
                "message": str(e.detail)
            })
            await websocket.close(code=1008)
            return
            
        except RateLimitExceeded as e:
            await websocket.send_json({
                "type": "error",
                "message": f"Rate limit exceeded: {str(e)}"
            })
            await websocket.close(code=1008)
            return
            
        except Exception as e:
            logger.error(f"Error during token validation: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": "Authentication failed"
            })
            await websocket.close(code=1008)
            return
            
    except WebSocketDisconnect:
        # Clean disconnection
        await manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Invalid JSON data"
            })
            await websocket.close(code=1003)
        except Exception:
            pass
            
    except Exception as e:
        logger.error(f"Unhandled exception in voice websocket: {str(e)}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Server error"
            })
            await websocket.close(code=1011)
        except Exception:
            pass

@app.get("/conversations", response_model=ConversationListResponse)
async def get_conversations(
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Get all conversations for a user"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id, increment=0.5)  # Lower increment for read
        
        # Get conversations
        conversations = await firestore_manager.get_conversations_by_user(user_id)
        
        # Sort by updated_at, newest first
        conversations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        # Return conversations
        return {
            "conversations": [
                {
                    "id": conv.get("id"),
                    "title": conv.get("title"),
                    "created_at": conv.get("created_at"),
                    "updated_at": conv.get("updated_at"),
                    "message_count": len(conv.get("messages", [])),
                    "type": conv.get("type", "text")
                }
                for conv in conversations
            ]
        }
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversations: {str(e)}"
        )

@app.get("/conversation/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Get a specific conversation"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id, increment=0.5)  # Lower increment for read
        
        # Get conversation
        conversation = await firestore_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Check ownership
        if conversation.get("user_id") != user_id:
            logger.warning(f"User {user_id} attempted to access conversation {conversation_id} owned by {conversation.get('user_id')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this conversation"
            )
        
        # Return conversation
        return {
            "id": conversation.get("id"),
            "title": conversation.get("title"),
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "messages": conversation.get("messages", [])
        }
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )

@app.put("/conversation", response_model=Dict[str, str])
async def update_conversation(
    request: UpdateConversationRequest,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Update a conversation title or add messages"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id)
        
        # Get conversation
        conversation_id = request.conversation_id
        conversation = await firestore_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Check ownership
        if conversation.get("user_id") != user_id:
            logger.warning(f"User {user_id} attempted to update conversation {conversation_id} owned by {conversation.get('user_id')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to update this conversation"
            )
        
        # Update title if provided
        if request.title is not None:
            conversation["title"] = request.title
        
        # Add messages if provided
        if request.messages:
            # Ensure messages have IDs and timestamps
            for msg in request.messages:
                if not msg.message_id:
                    msg.message_id = str(uuid.uuid4())
                if not msg.timestamp:
                    msg.timestamp = datetime.utcnow().isoformat()
            
            # Add new messages
            conversation["messages"].extend([msg.dict() for msg in request.messages])
        
        # Update timestamp
        conversation["updated_at"] = datetime.utcnow().isoformat()
        
        # Save to Firestore
        await firestore_manager.add_or_update_conversation(conversation_id, conversation)
        
        return {"message": "Conversation updated successfully"}
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error updating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update conversation: {str(e)}"
        )

@app.delete("/conversation", response_model=Dict[str, str])
async def delete_conversation(
    request: DeleteConversationRequest,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Delete a conversation"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id)
        
        # Get conversation
        conversation_id = request.conversation_id
        conversation = await firestore_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Check ownership
        if conversation.get("user_id") != user_id:
            logger.warning(f"User {user_id} attempted to delete conversation {conversation_id} owned by {conversation.get('user_id')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to delete this conversation"
            )
        
        # Delete conversation
        await firestore_manager.delete_conversation(conversation_id)
        
        return {"message": "Conversation deleted successfully"}
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    try:
        logger.info("API starting up")
        
        # Restore active connections from Firestore
        await manager.restore_from_firestore()
        
        logger.info("API startup complete")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    try:
        logger.info("API shutting down")
        
        # Close all WebSocket connections
        await manager.close_all()
        
        logger.info("API shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

# Function calling API
@app.post("/function-calling")
async def function_calling_endpoint(
    prompt: str = Body(..., embed=True),
    tools: List[Dict[str, Any]] = Body(..., embed=True),
    conversation_id: Optional[str] = Body(None, embed=True),
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Endpoint that supports function calling with the LLM"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id)
        
        # Get chat history if conversation ID provided
        chat_history = []
        if conversation_id:
            conversation = await firestore_manager.get_conversation(conversation_id)
            if conversation and conversation.get("user_id") == user_id:
                chat_history = conversation.get("messages", [])
        
        # Call LLM with function calling
        response = await llm_service.generate_with_function_calling(prompt, tools, chat_history)
        
        return response
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error in function calling: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Function calling error: {str(e)}"
        )

# Memory management API
@app.get("/memory/{conversation_id}")
async def get_memory(
    conversation_id: str,
    token_data: Dict[str, Any] = Depends(validate_device_token)
):
    """Get memory for a conversation"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id, increment=0.5)
        
        # Get memory
        memory = await memory_extractor.get_conversation_memory(conversation_id, user_id)
        
        return {"memory": memory}
        
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error getting memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get memory: {str(e)}"
        )

@app.post("/memory/extract/{conversation_id}")
async def extract_memory(
    conversation_id: str,
    token_data: Dict[str, Any] = Depends(validate_device_token),
    background_tasks: BackgroundTasks = None
):
    """Trigger memory extraction for a conversation"""
    try:
        user_id = token_data.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device not associated with a user"
            )
        
        # Rate limiting
        await rate_limiter.check_rate_limit(user_id)
        
        # Check conversation ownership
        conversation = await firestore_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        if conversation.get("user_id") != user_id:
            logger.warning(f"User {user_id} attempted to extract memory for conversation {conversation_id} owned by {conversation.get('user_id')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this conversation"
            )
        
        # Run extraction in background if background_tasks provided
        if background_tasks:
            background_tasks.add_task(extract_memory_task, conversation_id, user_id)
            return {"message": "Memory extraction started"}
        else:
            # Run synchronously if no background_tasks
            result = await memory_extractor.extract_key_info(conversation_id, user_id)
            return {
                "message": "Memory extraction completed",
                "result": result
            }
            
    except RateLimitExceeded as e:
        logger.warning(f"Rate limit exceeded for {token_data.get('user_id', token_data['device_id'])}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {str(e)}"
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"Error extracting memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract memory: {str(e)}"
        )