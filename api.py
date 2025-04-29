"""
Main REST API endpoints for EVA backend conversation.

Provides a standard HTTP endpoint for interacting with the conversation handler.
WebSocket interactions are handled by websocket_manager.py.
"""

import asyncio
import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

# --- Local Imports ---
from auth import get_current_active_user  # Use standard auth dependency
from conversation_handler import ConversationHandler  # Import the handler
from models import User
# Import exceptions if needed for specific handling
from exceptions import LLMServiceError, RateLimitError

# --- Router Setup ---
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# --- Request/Response Models ---
class ConversationRequest(BaseModel):
    """Request model for the REST conversation endpoint."""
    message: str
    session_id: Optional[str] = None  # Allow client to provide session ID if resuming
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata

class ConversationResponse(BaseModel):
    """Response model for the REST conversation endpoint."""
    response: str
    session_id: str
    function_calls: Optional[List[Dict[str, Any]]] = None  # Include if function call occurred
    error: Optional[str] = None  # Include errors if any

# --- REST Endpoint ---
@router.post(
    "/",  # Keep it at the root of the /api/v1/conversation prefix defined in main.py
    response_model=ConversationResponse,
    summary="Send a message and get a response (non-streaming)"
)
async def post_conversation(
    request_body: ConversationRequest,
    request: Request,  # Inject request for logging/state if needed
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> ConversationResponse:
    """
    Processes a single user message via the ConversationHandler and returns
    the complete response. This is a non-streaming endpoint.
    """
    session_id = request_body.session_id or str(uuid.uuid4())  # Use provided or generate new
    handler = ConversationHandler(current_user, session_id)
    user_message = request_body.message

    logger.info(f"REST request (Session: {session_id}): User {current_user.id} message: '{user_message[:50]}...'")

    full_response_text = ""
    function_calls_list = []
    error_message = None

    try:
        # Process the message using the handler's generator
        response_generator = handler.process_message(user_message)
        async for chunk in response_generator:
            if "text" in chunk:
                full_response_text += chunk["text"]
            elif "function_call" in chunk:
                # Store function calls encountered
                function_calls_list.append(chunk["function_call"])
                # NOTE: In a REST context, we typically return the function call info
                # and expect the client to make another request to execute it or provide results.
                # We won't execute it automatically here.
                logger.info(
                    f"REST request (Session: {session_id}): "
                    f"Function call requested: {chunk['function_call']['name']}"
                )
            elif "error" in chunk:
                # Capture the first error encountered
                if not error_message:
                    error_message = chunk["error"]
                logger.error(
                    f"REST request (Session: {session_id}): "
                    f"Error during processing: {chunk['error']}"
                )
                break  # Stop on first error

            # If this chunk is marked final, break out
            if chunk.get("is_final"):
                break

    except (LLMServiceError, RateLimitError) as e:
        logger.error(f"REST request (Session: {session_id}): Service Error: {e}", exc_info=True)
        error_message = f"Service Error: {e}"
    except Exception as e:
        logger.exception(f"REST request (Session: {session_id}): Unexpected Error: {e}", exc_info=True)
        error_message = "An unexpected server error occurred."

    # --- Construct Final Response ---
    if error_message:
        return ConversationResponse(
            response="",  # No successful response text
            session_id=session_id,
            error=error_message
        )
    return ConversationResponse(
        response=full_response_text,
        session_id=session_id,
        function_calls=function_calls_list or None,
        error=None
    )

# Note: The WebSocket endpoint previously in this file has been moved to websocket_manager.py