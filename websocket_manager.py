"""
WebSocket Manager for EVA backend.

This module provides WebSocket functionality for real-time voice streaming
and conversational interactions with the Gemini API.


Version 3 working
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from auth import get_current_user
from config import get_settings
from exceptions import AuthenticationError
from llm_service import GeminiService, process_audio_stream
from models import User
from api_tools import execute_function_call, available_tools, ToolCall

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# Initialize services
gemini_service = GeminiService()

# Active connections store
active_connections: Dict[str, WebSocket] = {}
user_sessions: Dict[str, Dict[str, Any]] = {}


class WebSocketMessage(BaseModel):
    """
    WebSocket message model.
    
    Attributes:
        message_type: Type of message (text, audio, command)
        content: Message content
        session_id: Unique session identifier
        metadata: Optional metadata for the message
    """
    message_type: str = Field(..., description="Type of message: text, audio, command")
    content: Any
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class StreamingResponse(BaseModel):
    """
    Streaming response model.
    
    Attributes:
        response_type: Type of response (text, audio, error, function_call)
        content: Response content
        session_id: Session identifier
        is_final: Whether this is the final response chunk
        function_calls: Optional function calls from LLM
    """
    response_type: str
    content: Any
    session_id: str
    is_final: bool = False
    function_calls: Optional[List[ToolCall]] = None


@asynccontextmanager
async def lifespan():
    """
    WebSocket connection manager lifespan handler.
    
    Handles cleanup of connections on shutdown.
    """
    yield
    # Clean up connections on shutdown
    for session_id, websocket in active_connections.items():
        await websocket.close(code=1000, reason="Server shutdown")
    active_connections.clear()
    user_sessions.clear()
    logger.info("All WebSocket connections closed on shutdown")


async def authenticate_websocket(websocket: WebSocket) -> Optional[User]:
    """
    Authenticate a WebSocket connection.
    
    Args:
        websocket: WebSocket connection
        
    Returns:
        Optional[User]: Authenticated user or None if authentication fails
        
    Raises:
        AuthenticationError: If authentication fails
    """
    # Get token from query parameters or headers
    token = websocket.query_params.get("token")
    if not token:
        # Try to get from headers
        headers = dict(websocket.headers)
        auth_header = headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
    
    if not token:
        logger.warning("WebSocket connection attempt without token")
        raise AuthenticationError("Missing authentication token")
    
    try:
        # Verify token and get user
        user = await get_current_user(token)
        logger.info(f"WebSocket authenticated for user: {user.username}")
        return user
    except Exception as e:
        logger.warning(f"WebSocket authentication failed: {str(e)}")
        raise AuthenticationError(f"Authentication failed: {str(e)}")


@router.websocket("/connect")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket connection endpoint.
    
    Handles real-time communication with clients for voice streaming
    and interactive conversations.
    
    Args:
        websocket: WebSocket connection
    """
    # Generate unique session ID
    session_id = str(uuid.uuid4())
    
    try:
        # Authenticate connection
        user = await authenticate_websocket(websocket)
        await websocket.accept()
        
        # Store connection
        active_connections[session_id] = websocket
        user_sessions[session_id] = {
            "user": user,
            "conversation_history": [],
            "active_stream": None
        }
        
        # Send initial connection success message
        await websocket.send_json({
            "response_type": "connection",
            "content": "Connected successfully",
            "session_id": session_id,
            "user_id": user.id
        })
        
        logger.info(f"WebSocket connection established for user {user.username}, session {session_id}")
        
        # Main message handling loop
        while True:
            # Receive message
            data = await websocket.receive()
            
            # Handle different message types
            if "text" in data:
                # Text message
                await handle_text_message(data["text"], session_id, websocket, user)
            elif "bytes" in data:
                # Binary audio data
                await handle_audio_stream(data["bytes"], session_id, websocket, user)
            else:
                # Parse as JSON for structured messages
                try:
                    message_data = json.loads(data.get("text", "{}"))
                    message = WebSocketMessage(**message_data)
                    await handle_structured_message(message, session_id, websocket, user)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message received: {data}")
                    await send_error(websocket, "Invalid message format", session_id)
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")
                    await send_error(websocket, f"Error processing message: {str(e)}", session_id)
    
    except AuthenticationError as e:
        logger.warning(f"WebSocket authentication failed: {str(e)}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e))
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected, session {session_id}")
        # Clean up on disconnect
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in user_sessions:
            del user_sessions[session_id]
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Internal server error")
        except:
            pass
        
        # Clean up on error
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in user_sessions:
            del user_sessions[session_id]


async def handle_text_message(
    text: str, 
    session_id: str, 
    websocket: WebSocket, 
    user: User
):
    """
    Handle text message from WebSocket.
    
    Args:
        text: Text message content
        session_id: Session identifier
        websocket: WebSocket connection
        user: Authenticated user
    """
    logger.info(f"Text message received in session {session_id}: {text[:50]}...")
    
    try:
        # Store in conversation history
        session_data = user_sessions.get(session_id, {})
        history = session_data.get("conversation_history", [])
        history.append({"role": "user", "content": text})
        
        # Generate streaming response from LLM
        response_generator = gemini_service.generate_streaming_response(
            text, 
            user.id, 
            conversation_history=history,
            tools=available_tools()
        )
        
        # Track full response for history
        full_response = ""
        function_calls = []
        
        # Stream response chunks to client
        async for chunk in response_generator:
            if chunk.get("type") == "text":
                text_chunk = chunk.get("text", "")
                full_response += text_chunk
                
                await websocket.send_json({
                    "response_type": "text",
                    "content": text_chunk,
                    "session_id": session_id,
                    "is_final": False
                })
            
            elif chunk.get("type") == "function_call":
                function_call = ToolCall(**chunk.get("function_call"))
                function_calls.append(function_call)
        
        # Store assistant response in history
        history.append({"role": "assistant", "content": full_response})
        session_data["conversation_history"] = history
        
        # Execute function calls if present
        if function_calls:
            executed_results = []
            for call in function_calls:
                try:
                    result = await execute_function_call(call, user)
                    executed_results.append({
                        "name": call.function.name,
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Function call execution error: {str(e)}")
                    executed_results.append({
                        "name": call.function.name,
                        "error": str(e)
                    })
            
            # Send function call results
            await websocket.send_json({
                "response_type": "function_call",
                "content": executed_results,
                "session_id": session_id,
                "is_final": False,
                "function_calls": [fc.model_dump() for fc in function_calls]
            })
        
        # Send final message
        await websocket.send_json({
            "response_type": "text",
            "content": "",
            "session_id": session_id,
            "is_final": True
        })
        
        logger.info(f"Response sent to session {session_id}, length: {len(full_response)}")
    
    except Exception as e:
        logger.error(f"Error processing text message: {str(e)}")
        await send_error(websocket, f"Error processing message: {str(e)}", session_id)


async def handle_audio_stream(
    audio_chunk: bytes, 
    session_id: str, 
    websocket: WebSocket, 
    user: User
):
    """
    Handle streaming audio from WebSocket.
    
    Args:
        audio_chunk: Binary audio data chunk
        session_id: Session identifier
        websocket: WebSocket connection
        user: Authenticated user
    """
    try:
        session_data = user_sessions.get(session_id, {})
        active_stream = session_data.get("active_stream")
        
        # Initialize stream processor if not already active
        if active_stream is None:
            logger.info(f"Initializing new audio stream for session {session_id}")
            active_stream = await process_audio_stream(
                user_id=user.id,
                callback=lambda response: send_stream_response(websocket, response, session_id)
            )
            session_data["active_stream"] = active_stream
        
        # Process audio chunk
        await active_stream.process_chunk(audio_chunk)
        
        # Check for end of audio marker (empty chunk or very small)
        if len(audio_chunk) < 10:  # Arbitrary small size to detect end
            logger.info(f"End of audio stream detected for session {session_id}")
            # Finalize stream processing
            result = await active_stream.finalize()
            
            # Clear active stream
            session_data["active_stream"] = None
            
            # Process any recognized commands
            if result.get("commands"):
                for command in result["commands"]:
                    # Handle command with function calling
                    function_calls = gemini_service.generate_function_calls(
                        command, 
                        user.id,
                        tools=available_tools()
                    )
                    
                    if function_calls:
                        # Execute function calls
                        for call in function_calls:
                            try:
                                result = await execute_function_call(call, user)
                                # Send function call result
                                await websocket.send_json({
                                    "response_type": "function_call",
                                    "content": {
                                        "name": call.function.name,
                                        "result": result
                                    },
                                    "session_id": session_id,
                                    "is_final": False
                                })
                            except Exception as e:
                                logger.error(f"Function call execution error: {str(e)}")
            
            # Send final message
            await websocket.send_json({
                "response_type": "audio_complete",
                "content": result.get("transcript", ""),
                "session_id": session_id,
                "is_final": True
            })
    
    except Exception as e:
        logger.error(f"Error processing audio stream: {str(e)}")
        await send_error(websocket, f"Error processing audio: {str(e)}", session_id)


async def handle_structured_message(
    message: WebSocketMessage, 
    session_id: str, 
    websocket: WebSocket, 
    user: User
):
    """
    Handle structured message from WebSocket.
    
    Args:
        message: Structured message
        session_id: Session identifier
        websocket: WebSocket connection
        user: Authenticated user
    """
    try:
        # Override session_id if provided in message
        if message.session_id:
            session_id = message.session_id
        
        if message.message_type == "text":
            await handle_text_message(message.content, session_id, websocket, user)
        
        elif message.message_type == "command":
            logger.info(f"Command received in session {session_id}: {message.content}")
            
            # Special command handling
            if message.content == "clear_history":
                # Clear conversation history
                user_sessions[session_id]["conversation_history"] = []
                await websocket.send_json({
                    "response_type": "command_result",
                    "content": "Conversation history cleared",
                    "session_id": session_id,
                    "is_final": True
                })
            
            elif message.content == "end_session":
                # End the session
                if session_id in user_sessions:
                    del user_sessions[session_id]
                await websocket.send_json({
                    "response_type": "command_result",
                    "content": "Session ended",
                    "session_id": session_id,
                    "is_final": True
                })
                await websocket.close(code=1000, reason="Session ended by client")
            
            else:
                # Process as generic command
                # This could integrate with the function calling system
                await websocket.send_json({
                    "response_type": "command_result",
                    "content": f"Command '{message.content}' received",
                    "session_id": session_id,
                    "is_final": True
                })
        
        else:
            logger.warning(f"Unknown message type: {message.message_type}")
            await send_error(
                websocket, 
                f"Unknown message type: {message.message_type}", 
                session_id
            )
    
    except Exception as e:
        logger.error(f"Error handling structured message: {str(e)}")
        await send_error(websocket, f"Error: {str(e)}", session_id)


async def send_stream_response(
    websocket: WebSocket, 
    response: Dict[str, Any], 
    session_id: str
):
    """
    Send streaming response to client.
    
    This is a callback function used by audio processing.
    
    Args:
        websocket: WebSocket connection
        response: Response data
        session_id: Session identifier
    """
    try:
        stream_response = StreamingResponse(
            response_type=response.get("type", "text"),
            content=response.get("content", ""),
            session_id=session_id,
            is_final=response.get("is_final", False)
        )
        
        await websocket.send_json(stream_response.model_dump())
    except Exception as e:
        logger.error(f"Error sending stream response: {str(e)}")


async def send_error(
    websocket: WebSocket, 
    error_message: str, 
    session_id: str
):
    """
    Send error message to client.
    
    Args:
        websocket: WebSocket connection
        error_message: Error message text
        session_id: Session identifier
    """
    try:
        await websocket.send_json({
            "response_type": "error",
            "content": error_message,
            "session_id": session_id,
            "is_final": True
        })
    except Exception as e:
        logger.error(f"Error sending error message: {str(e)}")