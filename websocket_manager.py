"""
WebSocket Manager for EVA backend.

Handles real-time WebSocket connections for chat and audio streaming,
integrating with ConversationHandler for processing and response generation.
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

# --- Local Imports ---
from auth import get_current_user, validate_request_auth # Use generalized auth validation
from config import get_settings
# Import the handler
from conversation_handler import ConversationHandler
# Import LLM service functions used directly (audio)
from llm_service import transcribe_audio, generate_voice_response # Keep direct audio functions for now
from models import User
from exceptions import AuthenticationError # Import specific exception

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# --- Connection Management ---
# Use dictionaries keyed by session_id for better tracking
active_connections: Dict[str, WebSocket] = {}
# Store handler instances associated with sessions
active_sessions: Dict[str, ConversationHandler] = {}

# --- WebSocket Message Models ---
class WebSocketMessage(BaseModel):
    """Incoming WebSocket message structure."""
    message_type: str = Field(..., description="Type of message: text, audio, command")
    content: Any
    session_id: Optional[str] = None # Client might send existing session ID to resume
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class StreamingResponse(BaseModel):
    """Outgoing WebSocket message structure."""
    response_type: str # e.g., "text_chunk", "full_text", "function_call", "error", "info", "transcription"
    content: Any
    session_id: str
    is_final: bool = False # Indicates the end of a logical response sequence
    # Add other fields as needed, e.g., for function calls
    function_call_info: Optional[Dict[str, Any]] = None


# --- Helper Functions ---

async def authenticate_websocket(websocket: WebSocket) -> Optional[User]:
    """Authenticates WebSocket connection using token from query param or header."""
    token: Optional[str] = None
    # 1. Check query parameter
    token = websocket.query_params.get("token")

    # 2. Check Authorization header if no query param
    if not token:
        auth_header = websocket.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1]

    # 3. Check Sec-WebSocket-Protocol (less common but possible)
    if not token:
         protocols = websocket.headers.get("sec-websocket-protocol", "").split(',')
         for proto in protocols:
              proto = proto.strip()
              if proto.lower().startswith("bearer_"):
                   token = proto.split('_', 1)[1]
                   break # Found token

    if not token:
        logger.warning("WebSocket connection attempt without token.")
        return None

    try:
        # Use the same validation as HTTP requests for consistency
        # Need to simulate a request object or adapt validate_request_auth
        # Simplified: directly use get_current_user which expects token string
        user = await get_current_user(token)
        if user and not user.disabled:
            logger.info(f"WebSocket authenticated for user: {user.id} ({user.username})")
            return user
        elif user:
            logger.warning(f"WebSocket authentication failed: User {user.id} is disabled.")
            return None
        else:
             # get_current_user raises exception on invalid token/user not found
             return None
    except AuthenticationError as e:
        logger.warning(f"WebSocket authentication failed: {e.detail}")
        return None
    except Exception as e:
        logger.error(f"Unexpected WebSocket authentication error: {e}", exc_info=True)
        return None

async def send_websocket_response(websocket: WebSocket, response_data: StreamingResponse):
    """Sends a structured response over the WebSocket."""
    try:
        await websocket.send_json(response_data.model_dump(exclude_none=True))
    except WebSocketDisconnect:
        # Handle disconnect during send if necessary, though usually caught in main loop
        logger.warning(f"Attempted to send to disconnected WebSocket (Session: {response_data.session_id})")
    except Exception as e:
        logger.error(f"Error sending WebSocket response (Session: {response_data.session_id}): {e}", exc_info=True)

async def close_websocket_session(session_id: str, code: int = status.WS_1000_NORMAL_CLOSURE, reason: str = "Session ended"):
    """Closes a WebSocket connection and cleans up resources."""
    websocket = active_connections.pop(session_id, None)
    session_handler = active_sessions.pop(session_id, None) # Changed variable name
    if websocket:
        try:
            await websocket.close(code=code, reason=reason)
            logger.info(f"WebSocket connection closed for session {session_id}. Reason: {reason}")
        except RuntimeError as e:
             # Catch errors if socket is already closed
             logger.warning(f"Error closing WebSocket for session {session_id} (possibly already closed): {e}")
        except Exception as e:
            logger.error(f"Unexpected error closing WebSocket for session {session_id}: {e}", exc_info=True)
    if session_handler:
         # Perform any cleanup needed for the handler/context
         # e.g., session_handler.context_window.save_state() if needed
         logger.debug(f"Cleaned up handler for session {session_id}")


# --- WebSocket Endpoints ---

@router.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """Handles real-time text-based chat conversations via WebSocket."""
    user = await authenticate_websocket(websocket)
    if not user:
        # Close unauthorized connection immediately
        await websocket.accept() # Must accept before closing with code
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()

    # Initialize session
    session_id = str(uuid.uuid4())
    handler = ConversationHandler(user, session_id)
    active_connections[session_id] = websocket
    active_sessions[session_id] = handler # Store handler instance
    logger.info(f"Chat WebSocket session {session_id} started for user {user.id}")

    # Send session ID confirmation
    await send_websocket_response(websocket, StreamingResponse(
        response_type="info",
        content={"message": "Session started.", "session_id": session_id},
        session_id=session_id,
        is_final=True # This specific info message is final
    ))

    try:
        while True:
            raw_data = await websocket.receive_text() # Use receive_text for flexibility, parse JSON manually
            try:
                data = json.loads(raw_data)
                message = WebSocketMessage(**data)
                logger.debug(f"Received message (Session: {session_id}): Type={message.message_type}")

                if message.message_type == "text":
                    if not message.content or not isinstance(message.content, str):
                         raise ValueError("Text message content cannot be empty.")

                    # Use the ConversationHandler's generator
                    response_generator = handler.process_message(message.content)
                    async for chunk in response_generator:
                        response_type = "text_chunk" # Default for text
                        is_final = chunk.get("is_final", False)
                        content = None
                        func_call_info = None

                        if "text" in chunk:
                            content = chunk["text"]
                            # Could differentiate final text chunk type if desired
                            # if is_final: response_type = "full_text"
                        elif "function_call" in chunk:
                            response_type = "function_call"
                            content = f"Function call requested: {chunk['function_call']['name']}" # Info for client
                            func_call_info = chunk['function_call']
                            # Decide if function call implies is_final=True for this turn
                            # is_final = True # Or based on chunk flag
                        elif "error" in chunk:
                            response_type = "error"
                            content = chunk["error"]
                            is_final = True # Errors are usually final for the turn

                        if content is not None:
                            await send_websocket_response(websocket, StreamingResponse(
                                response_type=response_type,
                                content=content,
                                session_id=session_id,
                                is_final=is_final,
                                function_call_info=func_call_info
                            ))

                elif message.message_type == "command":
                    # Handle client-side commands like clear context, end session
                    command = message.content
                    if command == "end_session":
                         logger.info(f"Client requested end_session for session {session_id}")
                         await close_websocket_session(session_id, reason="Client ended session")
                         break # Exit loop
                    # Add other commands as needed
                    # elif command == "clear_context":
                    #     handler.context_window.clear() ... send confirmation
                    else:
                         await send_websocket_response(websocket, StreamingResponse(
                              response_type="error", content=f"Unknown command: {command}", session_id=session_id, is_final=True
                         ))

                # Add handling for other message types if needed (e.g., function_result)

            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON via WebSocket (Session: {session_id})")
                await send_websocket_response(websocket, StreamingResponse(
                    response_type="error", content="Invalid JSON message format.", session_id=session_id, is_final=True
                ))
            except Exception as e: # Catch errors during message processing
                logger.exception(f"Error processing WebSocket message (Session: {session_id}): {e}", exc_info=True)
                await send_websocket_response(websocket, StreamingResponse(
                    response_type="error", content=f"An error occurred: {e}", session_id=session_id, is_final=True
                ))

    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket disconnected (Session: {session_id})")
    except Exception as e:
        logger.exception(f"Unexpected error in chat WebSocket loop (Session: {session_id}): {e}", exc_info=True)
    finally:
        # Ensure cleanup happens regardless of how the loop exits
        await close_websocket_session(session_id, reason="Connection closed or error")


@router.websocket("/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """Handles real-time audio streaming, transcription, and voice response."""
    user = await authenticate_websocket(websocket)
    if not user:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()

    session_id = str(uuid.uuid4())
    handler = ConversationHandler(user, session_id) # Use the same handler
    active_connections[session_id] = websocket
    active_sessions[session_id] = handler
    logger.info(f"Audio WebSocket session {session_id} started for user {user.id}")

    await send_websocket_response(websocket, StreamingResponse(
        response_type="info",
        content={"message": "Audio session started.", "session_id": session_id},
        session_id=session_id,
        is_final=True
    ))

    audio_buffer = bytearray()
    # Add state for VAD (Voice Activity Detection) if implementing chunking

    try:
        while True:
            # Receive binary audio data or control messages (JSON)
            raw_data = await websocket.receive_bytes() # Primarily expect bytes for audio

            # Basic check: If data looks like JSON, treat as control message
            try:
                 # Attempt to decode as UTF-8 and parse JSON
                 control_msg_str = raw_data.decode('utf-8')
                 control_msg_data = json.loads(control_msg_str)
                 message = WebSocketMessage(**control_msg_data)

                 if message.message_type == "command":
                      if message.content == "end_audio_stream":
                           logger.info(f"Client signaled end of audio stream for session {session_id}. Processing buffer...")
                           # Process any remaining audio in the buffer
                           if audio_buffer:
                                # --- Process buffered audio ---
                                try:
                                    transcribed_text = await transcribe_audio(bytes(audio_buffer))
                                    audio_buffer.clear() # Clear buffer after processing
                                    if not transcribed_text: raise ValueError("Transcription failed or empty.")

                                    await send_websocket_response(websocket, StreamingResponse(
                                        response_type="transcription", content=transcribed_text, session_id=session_id, is_final=False
                                    ))

                                    # Process text with conversation handler (streaming response)
                                    response_generator = handler.process_message(transcribed_text)
                                    async for chunk in response_generator:
                                        # ... (send chunks as in chat endpoint) ...
                                        response_type = "text_chunk"
                                        is_final = chunk.get("is_final", False)
                                        content = chunk.get("text") or chunk.get("error") # Prioritize text/error
                                        func_call_info = chunk.get("function_call")

                                        if content:
                                            await send_websocket_response(websocket, StreamingResponse(
                                                response_type="error" if "error" in chunk else response_type,
                                                content=content, session_id=session_id, is_final=is_final,
                                                function_call_info=func_call_info
                                            ))
                                except Exception as audio_proc_err:
                                     logger.error(f"Error processing buffered audio (Session: {session_id}): {audio_proc_err}", exc_info=True)
                                     await send_websocket_response(websocket, StreamingResponse(
                                          response_type="error", content=f"Error processing audio: {audio_proc_err}", session_id=session_id, is_final=True
                                     ))
                           else:
                                logger.info(f"End of audio stream signal received, buffer empty (Session: {session_id}).")
                           # Optionally wait for final response processing before breaking?
                           # For now, break after processing buffer.

                      elif message.content == "end_session":
                           logger.info(f"Client requested end_session via audio socket for session {session_id}")
                           await close_websocket_session(session_id, reason="Client ended session")
                           break # Exit loop
                      else:
                           logger.warning(f"Received unknown command via audio socket: {message.content}")
                 continue # Skip audio processing if it was a control message

            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                 # Assume it's binary audio data if decoding/parsing fails
                 audio_buffer.extend(raw_data)
                 logger.debug(f"Received audio chunk: {len(raw_data)} bytes. Buffer size: {len(audio_buffer)} (Session: {session_id})")
                 # TODO: Implement VAD or chunking logic here.
                 # If a complete utterance is detected (e.g., silence after speech),
                 # process the audio_buffer like in the "end_audio_stream" block above.
                 # For now, we only process on explicit "end_audio_stream" command.

    except WebSocketDisconnect:
        logger.info(f"Audio WebSocket disconnected (Session: {session_id})")
        # Process any remaining buffer on disconnect? Optional.
        if audio_buffer:
             logger.info(f"Processing remaining audio buffer ({len(audio_buffer)} bytes) on disconnect (Session: {session_id}).")
             # Add processing logic here if desired
    except Exception as e:
        logger.exception(f"Unexpected error in audio WebSocket loop (Session: {session_id}): {e}", exc_info=True)
    finally:
        await close_websocket_session(session_id, reason="Connection closed or error")
