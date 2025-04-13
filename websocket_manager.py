"""
WebSocket Manager for EVA backend.

This module provides WebSocket connections for real-time communication
with the EVA backend, including streaming responses and memory integration.

Update your existing websocket_manager.py with this enhanced version.

Current Date: 2025-04-13
Current User: IAmLep
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, Field

from auth import get_current_user
from config import get_settings
from context_window import get_context_window
from memory_manager import get_memory_manager, MemoryType, MemoryCommand
from llm_service import GeminiService, process_audio_stream, generate_voice_response
from models import User

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# Active connections store
active_connections: Dict[str, WebSocket] = {}
user_sessions: Dict[str, Dict[str, Any]] = {}


class WebSocketMessage(BaseModel):
    """
    WebSocket message model.
    
    Attributes:
        message_type: Type of message: text, audio, command
        content: Message content
        session_id: Optional session identifier
        metadata: Optional additional metadata
    """
    message_type: str = Field(..., description="Type of message: text, audio, command")
    content: Any
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class StreamingResponse(BaseModel):
    """
    Streaming response model.
    
    Attributes:
        response_type: Type of response: text, audio, error, etc.
        content: Response content
        session_id: Session identifier
        is_final: Whether this is the final chunk
        function_calls: Optional function calls
    """
    response_type: str
    content: Any
    session_id: str
    is_final: bool = False
    function_calls: Optional[List[Dict[str, Any]]] = None


class ConversationSession:
    """
    Conversation session manager for WebSocket connections.
    
    Manages the state of an ongoing conversation with context and memory.
    """
    def __init__(self, user: User, session_id: Optional[str] = None):
        """
        Initialize a conversation session.
        
        Args:
            user: The user who owns this session
            session_id: Optional session ID (generated if not provided)
        """
        self.user = user
        self.session_id = session_id or str(uuid.uuid4())
        self.context_window = get_context_window()
        self.memory_manager = get_memory_manager()
        self.gemini_service = GeminiService()
        self.start_time = datetime.utcnow()
        self.last_activity = self.start_time
        self.message_count = 0
        
        # Add system instructions
        self._add_system_instructions()
        
        logger.info(f"Created conversation session {self.session_id} for user {user.id}")
    
    def _add_system_instructions(self):
        """Add system instructions to the context window."""
        # Add persona instruction
        self.context_window.add_system_instruction(
            "You are EVA, an advanced AI assistant designed to be helpful, accurate, and conversational. "
            "You have a friendly, supportive personality and should respond in a natural, human-like way. "
            "Always maintain a helpful, positive tone while remaining truthful and informative."
        )
        
        # Add current date/time instruction
        current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        self.context_window.add_system_instruction(
            f"Current date and time: {current_time}"
        )
        
        # Add user-specific instruction
        self.context_window.add_system_instruction(
            f"You are speaking with {self.user.username}. Refer to them by name occasionally "
            f"to personalize the conversation."
        )
    
    async def process_message(self, message: str) -> Optional[MemoryCommand]:
        """
        Process a user message and update the context.
        
        Args:
            message: User message
            
        Returns:
            Optional[MemoryCommand]: Memory command if detected
        """
        # Check for memory commands
        memory_command = await self.memory_manager.extract_memory_command(message)
        
        # Add message to context
        self.context_window.add_message("user", message)
        self.message_count += 1
        self.last_activity = datetime.utcnow()
        
        return memory_command
    
    async def get_response(
        self, 
        stream: bool = True,
        websocket: Optional[WebSocket] = None
    ) -> str:
        """
        Get response for the current context.
        
        Args:
            stream: Whether to stream the response
            websocket: Optional WebSocket for streaming
            
        Returns:
            str: Generated response
        """
        try:
            # Refresh relevant memories
            await self.context_window.refresh_memories(
                self.user.id,
                # Use the last few messages to find relevant memories
                " ".join([m.content for m in self.context_window.recent_messages[-3:]])
            )
            
            # Assemble context
            context = self.context_window.assemble_context()
            
            if stream and websocket:
                # Stream the response
                full_response = ""
                async for chunk in self.gemini_service.stream_conversation(context):
                    full_response += chunk
                    await send_stream_response(
                        websocket,
                        chunk,
                        self.session_id,
                        False
                    )
                
                # Send final message
                await send_stream_response(
                    websocket,
                    full_response,
                    self.session_id,
                    True
                )
                
                # Add assistant response to context
                self.context_window.add_message("assistant", full_response)
                return full_response
            else:
                # Generate complete response
                response, _, _ = await self.gemini_service.generate_text(context)
                
                # Add assistant response to context
                self.context_window.add_message("assistant", response)
                return response
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    async def handle_memory_command(self, command: MemoryCommand) -> str:
        """
        Handle a memory command.
        
        Args:
            command: Memory command to execute
            
        Returns:
            str: Response confirming the action
        """
        try:
            if command.command_type == "remember":
                # Create a core memory
                category = command.category or MemoryType.CORE
                await self.memory_manager.create_core_memory(
                    user_id=self.user.id,
                    content=command.content,
                    category=category,
                    entity=command.entity
                )
                return f"I'll remember that: {command.content}"
                
            elif command.command_type == "remind":
                # Create an event memory
                if command.event_time:
                    await self.memory_manager.create_event_memory(
                        user_id=self.user.id,
                        content=command.content,
                        event_time=command.event_time,
                        expiration=command.expiration
                    )
                    event_time_str = command.event_time.strftime("%Y-%m-%d %H:%M")
                    return f"I'll remind you at {event_time_str}: {command.content}"
                else:
                    return "I couldn't determine the time for your reminder. Please specify a time."
                
            elif command.command_type == "forget":
                # This would require searching for the memory first
                return "I'm not sure which memory to forget. Could you be more specific?"
                
            return "I'm not sure how to process that command."
            
        except Exception as e:
            logger.error(f"Error handling memory command: {str(e)}")
            return f"I encountered an error processing your request: {str(e)}"


async def authenticate_websocket(websocket: WebSocket) -> Optional[User]:
    """
    Authenticate WebSocket connection.
    
    Args:
        websocket: WebSocket connection
        
    Returns:
        Optional[User]: Authenticated user or None if authentication failed
    """
    try:
        # Get token from query parameters or headers
        token = websocket.query_params.get("token")
        if not token and "authorization" in websocket.headers:
            auth_header = websocket.headers["authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        if not token:
            return None
            
        # Validate token and get user
        from auth import get_current_user
        user = await get_current_user(token)
        return user
    except Exception as e:
        logger.error(f"WebSocket authentication error: {str(e)}")
        return None


async def send_stream_response(
    websocket: WebSocket,
    content: str,
    session_id: str,
    is_final: bool = False
) -> None:
    """
    Send streaming response to WebSocket.
    
    Args:
        websocket: WebSocket connection
        content: Response content
        session_id: Session identifier
        is_final: Whether this is the final chunk
    """
    try:
        response = StreamingResponse(
            response_type="text",
            content=content,
            session_id=session_id,
            is_final=is_final
        )
        await websocket.send_json(response.dict())
    except Exception as e:
        logger.error(f"Error sending stream response: {str(e)}")


async def send_error(
    websocket: WebSocket,
    error_message: str,
    session_id: str
) -> None:
    """
    Send error message to WebSocket.
    
    Args:
        websocket: WebSocket connection
        error_message: Error message
        session_id: Session identifier
    """
    try:
        response = StreamingResponse(
            response_type="error",
            content=error_message,
            session_id=session_id,
            is_final=True
        )
        await websocket.send_json(response.dict())
    except Exception as e:
        logger.error(f"Error sending error message: {str(e)}")


@router.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat.
    
    This endpoint handles authentication, message processing, and streaming responses.
    """
    # Authenticate user
    user = await authenticate_websocket(websocket)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return
    
    # Accept connection
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for user {user.id}")
    
    # Create session or get existing one
    session_id = str(uuid.uuid4())
    session = ConversationSession(user, session_id)
    
    # Store connection
    active_connections[session_id] = websocket
    user_sessions[session_id] = session
    
    try:
        # Send initial greeting
        await send_stream_response(
            websocket,
            "Hello! How can I help you today?",
            session_id,
            True
        )
        
        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            try:
                # Parse message
                message = WebSocketMessage(**data)
                
                if message.message_type == "text":
                    # Process text message
                    memory_command = await session.process_message(message.content)
                    
                    if memory_command:
                        # Handle memory command
                        response = await session.handle_memory_command(memory_command)
                        await send_stream_response(websocket, response, session_id, True)
                    else:
                        # Get normal response
                        await session.get_response(stream=True, websocket=websocket)
                        
                elif message.message_type == "audio":
                    # Process audio message
                    try:
                        audio_data = base64.b64decode(message.content)
                        text = await process_audio_stream(audio_data)
                        
                        # Process the transcribed text
                        memory_command = await session.process_message(text)
                        
                        # Send transcription back to client
                        await send_stream_response(
                            websocket,
                            f"Transcription: {text}",
                            session_id,
                            False
                        )
                        
                        if memory_command:
                            response = await session.handle_memory_command(memory_command)
                            await send_stream_response(websocket, response, session_id, True)
                        else:
                            await session.get_response(stream=True, websocket=websocket)
                    except Exception as e:
                        logger.error(f"Audio processing error: {str(e)}")
                        await send_error(websocket, f"Failed to process audio: {str(e)}", session_id)
                
                elif message.message_type == "command":
                    # Process custom command
                    command = message.content
                    if command == "clear_context":
                        session.context_window.clear()
                        session._add_system_instructions()
                        await send_stream_response(
                            websocket,
                            "Context cleared. Starting a fresh conversation.",
                            session_id,
                            True
                        )
                    else:
                        await send_error(
                            websocket,
                            f"Unknown command: {command}",
                            session_id
                        )
                
                else:
                    await send_error(
                        websocket,
                        f"Unsupported message type: {message.message_type}",
                        session_id
                    )
                    
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await send_error(websocket, f"Error: {str(e)}", session_id)
    
    except WebSocketDisconnect:
        # Handle disconnection
        logger.info(f"WebSocket disconnected for session {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in user_sessions:
            del user_sessions[session_id]
    
    except Exception as e:
        # Handle other errors
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await send_error(websocket, f"Server error: {str(e)}", session_id)
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass
        
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in user_sessions:
            del user_sessions[session_id]


@router.websocket("/audio-stream")
async def audio_streaming_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for audio streaming.
    
    This endpoint handles real-time audio transcription and response generation.
    """
    # Similar authentication and session setup as the chat endpoint
    user = await authenticate_websocket(websocket)
    if not user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return
    
    # Accept connection
    await websocket.accept()
    logger.info(f"Audio WebSocket connection accepted for user {user.id}")
    
    # Create session
    session_id = str(uuid.uuid4())
    session = ConversationSession(user, session_id)
    
    # Handle audio streaming
    # This is a simplified implementation
    try:
        while True:
            # Receive audio chunk
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if "audio" in data:
                # Process audio chunk
                audio_data = base64.b64decode(data["audio"])
                
                # For simplicity, we're assuming complete audio in each message
                # In a real implementation, you might buffer chunks until complete
                try:
                    text = await process_audio_stream(audio_data)
                    
                    # Process the transcribed text
                    memory_command = await session.process_message(text)
                    
                    # Send transcription to client
                    await send_stream_response(
                        websocket,
                        f"Transcription: {text}",
                        session_id,
                        False
                    )
                    
                    # Generate and send response
                    if memory_command:
                        response = await session.handle_memory_command(memory_command)
                    else:
                        response = await session.get_response(stream=False)
                    
                    # Generate voice response if requested
                    if data.get("voice_response", False):
                        try:
                            audio_bytes = await generate_voice_response(response)
                            voice_response = StreamingResponse(
                                response_type="audio",
                                content=audio_bytes.decode(),
                                session_id=session_id,
                                is_final=True
                            )
                            await websocket.send_json(voice_response.dict())
                        except Exception as e:
                            logger.error(f"Voice generation error: {str(e)}")
                            # Fall back to text response
                            await send_stream_response(websocket, response, session_id, True)
                    else:
                        await send_stream_response(websocket, response, session_id, True)
                
                except Exception as e:
                    logger.error(f"Audio processing error: {str(e)}")
                    await send_error(websocket, f"Failed to process audio: {str(e)}", session_id)
            
            elif data.get("command") == "end_session":
                await send_stream_response(
                    websocket,
                    "Audio session ended",
                    session_id,
                    True
                )
                break
    
    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected for session {session_id}")
    
    except Exception as e:
        logger.error(f"Audio WebSocket error: {str(e)}")
        try:
            await send_error(websocket, f"Server error: {str(e)}", session_id)
        except:
            pass