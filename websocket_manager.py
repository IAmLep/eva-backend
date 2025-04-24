"""
WebSocket Manager for EVA backend.

Handles real-time WebSocket connections for chat and audio streaming,
integrating with ConversationHandler for processing and response generation,
including handling the function calling loop.
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
from auth import get_current_user # Removed validate_request_auth as it wasn't used
from config import settings
from conversation_handler import ConversationHandler
from llm_service import transcribe_audio, generate_voice_response # Keep direct audio functions for now
from models import User
from exceptions import AuthenticationError, FunctionCallError # Import specific exceptions
# Import tool execution function if server-side execution is intended
try:
    from api_tools import execute_function_call
    SERVER_EXECUTES_FUNCTIONS = True
except ImportError:
    SERVER_EXECUTES_FUNCTIONS = False
    execute_function_call = None # Define placeholder

# Setup router
router = APIRouter()
logger = logging.getLogger(__name__)

# Connection Management (Unchanged)
active_connections: Dict[str, WebSocket] = {}
active_sessions: Dict[str, ConversationHandler] = {}

# WebSocket Models (Unchanged)
class WebSocketMessage(BaseModel): ...
class StreamingResponse(BaseModel): ...

# Helper Functions (authenticate_websocket, send_websocket_response, close_websocket_session - Unchanged)
async def authenticate_websocket(websocket: WebSocket) -> Optional[User]: ...
async def send_websocket_response(websocket: WebSocket, response_data: StreamingResponse): ...
async def close_websocket_session(session_id: str, code: int = status.WS_1000_NORMAL_CLOSURE, reason: str = "Session ended"): ...


# --- WebSocket Endpoints ---

@router.websocket("/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    """Handles real-time text-based chat conversations via WebSocket."""
    user = await authenticate_websocket(websocket)
    if not user:
        await websocket.accept()
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    # Pass user object and session_id to handler
    handler = ConversationHandler(user=user, session_id=session_id)
    active_connections[session_id] = websocket
    active_sessions[session_id] = handler
    logger.info(f"Chat WebSocket session {session_id} started for user {user.id}")

    await send_websocket_response(websocket, StreamingResponse(
        response_type="info",
        content={"message": "Session started.", "session_id": session_id},
        session_id=session_id,
        is_final=True
    ))

    try:
        while True:
            raw_data = await websocket.receive_text()
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
                        # Determine response type based on chunk content
                        response_type = "unknown"
                        content = chunk
                        is_final = chunk.get("is_final", False)
                        func_call_info = None

                        if "text" in chunk and chunk["text"]:
                             response_type = "text_chunk"
                             content = chunk["text"]
                        elif "function_call_request" in chunk:
                             response_type = "function_call_request" # Inform client about the request
                             func_call_info = chunk["function_call_request"]
                             content = f"Requesting function call: {func_call_info['name']}"
                             # Server will execute, no further action needed from client immediately
                        elif "function_call_result" in chunk:
                             response_type = "function_call_result" # Inform client about the result
                             content = chunk["function_call_result"]
                             # Content contains the result summary/details
                        elif "error" in chunk:
                             response_type = "error"
                             content = chunk["error"]
                             is_final = True # Errors are usually final

                        # Send the chunk regardless of type
                        await send_websocket_response(websocket, StreamingResponse(
                            response_type=response_type,
                            content=content,
                            session_id=session_id,
                            is_final=is_final,
                            function_call_info=func_call_info # Send request details if applicable
                        ))

                        # If the generator finishes (is_final=True), break inner loop
                        if is_final:
                            break

                # --- Add Handling for Function Result from Client (if client executes) ---
                # elif message.message_type == "function_result" and not SERVER_EXECUTES_FUNCTIONS:
                #     if 'name' not in message.content or 'result' not in message.content:
                #         raise ValueError("Function result message missing 'name' or 'result'.")
                #     logger.info(f"Received function result for {message.content['name']} from client.")
                #     # Feed result back to handler
                #     result_generator = handler.process_function_result(
                #         function_name=message.content['name'],
                #         tool_result=message.content['result'] # Assuming result is JSON-serializable dict
                #     )
                #     async for chunk in result_generator:
                #          # Process and send subsequent chunks (text or error)
                #          # ... (similar logic as above) ...

                elif message.message_type == "command":
                    command = message.content
                    if command == "end_session":
                         await close_websocket_session(session_id, reason="Client ended session")
                         break # Exit outer loop
                    else:
                         await send_websocket_response(websocket, StreamingResponse(
                              response_type="error", content=f"Unknown command: {command}", session_id=session_id, is_final=True
                         ))

            except json.JSONDecodeError:
                await send_websocket_response(websocket, StreamingResponse(
                    response_type="error", content="Invalid JSON message format.", session_id=session_id, is_final=True
                ))
            except (ValueError, FunctionCallError) as e: # Catch specific known errors
                await send_websocket_response(websocket, StreamingResponse(
                    response_type="error", content=f"Processing error: {e}", session_id=session_id, is_final=True
                ))
            except Exception as e: # Catch unexpected errors
                logger.exception(f"Error processing WebSocket message (Session: {session_id}): {e}", exc_info=True)
                await send_websocket_response(websocket, StreamingResponse(
                    response_type="error", content=f"An unexpected server error occurred: {e}", session_id=session_id, is_final=True
                ))

    except WebSocketDisconnect:
        logger.info(f"Chat WebSocket disconnected (Session: {session_id})")
    except Exception as e:
        logger.exception(f"Unexpected error in chat WebSocket loop (Session: {session_id}): {e}", exc_info=True)
    finally:
        await close_websocket_session(session_id, reason="Connection closed or error")


@router.websocket("/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    # (Implementation largely unchanged, still uses process_message)
    # Ensure it handles the new chunk types ('function_call_request', 'function_call_result') appropriately if needed for audio flow.
    ...