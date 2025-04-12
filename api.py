"""
API module for EVA backend.

This module provides the main API endpoints for text-based communication
and function calling integration with Gemini API.

"""
"""
Version 3 working
"""

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from auth import get_current_active_user, validate_request_auth
from config import get_settings
from database import get_db_manager
from exceptions import AuthorizationError, DatabaseError
from llm_service import GeminiService, generate_response
from models import User, Memory
from api_tools import execute_function_call, available_tools, ToolCall

# Setup router
router = APIRouter()

# Logger configuration
logger = logging.getLogger(__name__)

# Initialize services
gemini_service = GeminiService()


class MessageRequest(BaseModel):
    """
    Message request model for API.
    
    Attributes:
        text: Message text content
        user_id: Optional ID of the user sending the message
        context_id: Optional context/conversation ID
        include_memory: Whether to include user memory in context
    """
    text: str
    user_id: Optional[str] = None
    context_id: Optional[str] = None
    include_memory: bool = True


class SimpleMessageRequest(BaseModel):
    """
    Simple message request for service account access.
    
    Attributes:
        message: Message text content
        context: Optional context for the message
    """
    message: str
    context: Optional[Dict[str, Any]] = None


class MessageResponse(BaseModel):
    """
    Message response model from API.
    
    Attributes:
        text: Response text content
        context_id: Context/conversation ID
        function_calls: Optional list of function calls from LLM
    """
    text: str
    context_id: str
    function_calls: Optional[List[ToolCall]] = None


class SimpleMessageResponse(BaseModel):
    """
    Simple message response model for service account access.
    
    Attributes:
        response: Response text content
        tokens: Token usage information
        model: Model used for generation
        timestamp: Response timestamp
    """
    response: str
    tokens: Dict[str, int] = Field(default_factory=dict)
    model: str = "gemini-2.0-flash"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class MemoryRequest(BaseModel):
    """
    Memory creation request.
    
    Attributes:
        content: Memory content
        metadata: Optional metadata for the memory
    """
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# Original complex message endpoint
@router.post("/message", response_model=MessageResponse)
async def process_message(
    message: MessageRequest,
    current_user: User = Depends(validate_request_auth)
) -> MessageResponse:
    """
    Process a text message and generate a response.
    
    Args:
        message: Message request containing text and context
        current_user: Current authenticated user from dependency
        
    Returns:
        MessageResponse: Generated response with possible function calls
        
    Raises:
        HTTPException: If message processing fails
        AuthorizationError: If user isn't authorized
    """
    # Use authenticated user ID if not provided in request
    user_id = message.user_id or current_user.username
    
    # For service accounts, skip user authorization check
    if not current_user.is_service_account and current_user.username != user_id:
        logger.warning(f"User {current_user.username} attempted to access messages for user {user_id}")
        raise AuthorizationError(detail="Not authorized to access this user's messages")
    
    try:
        # Get user memories if requested
        memories = []
        if message.include_memory:
            db = get_db_manager()
            # Get relevant memories that might help with this message
            memory_docs = await db.get_relevant_memories(user_id, message.text)
            memories = [Memory(**doc) for doc in memory_docs]
            logger.info(f"Retrieved {len(memories)} relevant memories for user {user_id}")
        
        # Generate response with LLM
        response_text, context_id, function_calls = await generate_response(
            message.text,
            user_id,
            message.context_id,
            memories,
            tools=available_tools()
        )
        
        # Execute function calls if present
        executed_calls = []
        if function_calls:
            for call in function_calls:
                try:
                    # Execute the function call
                    result = await execute_function_call(call, current_user)
                    logger.info(f"Executed function call: {call.function.name} for user {user_id}")
                    
                    # Append result to response if needed
                    if result.get("append_to_response", False):
                        response_text += f"\n\n{result.get('message', '')}"
                    
                    executed_calls.append(call)
                except Exception as e:
                    logger.error(f"Function call execution error: {str(e)}")
                    # Don't fail the whole request if a function call fails
        
        return MessageResponse(
            text=response_text,
            context_id=context_id,
            function_calls=executed_calls if executed_calls else None
        )
    
    except DatabaseError as e:
        logger.error(f"Database error during message processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


# Simple message endpoint for service accounts
@router.post("/simple-message", response_model=SimpleMessageResponse)
async def process_simple_message(
    request: SimpleMessageRequest,
    user: User = Depends(validate_request_auth)
) -> Dict:
    """
    Process a simple message without memory management.
    
    Args:
        request: Simple message request
        user: Authenticated user
        
    Returns:
        Dict: Simple message response
    """
    logger.info(f"Processing simple message request from {user.username}")
    
    try:
        # Get response from LLM service with minimal context
        response, token_info = await gemini_service.generate_text(
            request.message,
            temperature=0.7,
            max_tokens=800
        )
        
        # Ensure token_info is a dictionary even if None is returned
        if token_info is None:
            logger.warning("Received None for token_info, using default values")
            token_info = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        
        return {
            "response": response,
            "tokens": token_info,
            "model": "gemini-2.0-flash",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error processing simple message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )


# Add alias endpoints for compatibility
@router.post("/chat", response_model=SimpleMessageResponse)
async def process_chat(
    request: SimpleMessageRequest,
    user: User = Depends(validate_request_auth)
) -> Dict:
    """
    Process a chat message (alias for simple-message).
    """
    return await process_simple_message(request, user)


@router.post("/generate", response_model=SimpleMessageResponse)
async def generate_text(
    request: SimpleMessageRequest,
    user: User = Depends(validate_request_auth)
) -> Dict:
    """
    Generate text (alias for simple-message).
    """
    return await process_simple_message(request, user)


@router.post("/gemini/chat", response_model=SimpleMessageResponse)
async def gemini_chat(
    request: SimpleMessageRequest,
    user: User = Depends(validate_request_auth)
) -> Dict:
    """
    Gemini chat endpoint (alias for simple-message).
    """
    return await process_simple_message(request, user)


@router.post("/llm/generate", response_model=SimpleMessageResponse)
async def llm_generate(
    request: SimpleMessageRequest,
    user: User = Depends(validate_request_auth)
) -> Dict:
    """
    LLM generate endpoint (alias for simple-message).
    """
    return await process_simple_message(request, user)


@router.post("/memory", status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_request: MemoryRequest,
    current_user: User = Depends(validate_request_auth)
) -> Dict[str, str]:
    """
    Create a new memory for the user.
    
    Args:
        memory_request: Memory creation request
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Response with memory_id
        
    Raises:
        HTTPException: If memory creation fails
    """
    try:
        db = get_db_manager()
        memory = Memory(
            user_id=current_user.username,
            content=memory_request.content,
            metadata=memory_request.metadata
        )
        
        memory_id = await db.create_memory(memory)
        logger.info(f"Created memory {memory_id} for user {current_user.username}")
        
        return {"memory_id": memory_id}
    except Exception as e:
        logger.error(f"Error creating memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating memory: {str(e)}"
        )


@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    current_user: User = Depends(validate_request_auth)
) -> Dict[str, bool]:
    """
    Delete a user memory.
    
    Args:
        memory_id: ID of memory to delete
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Response with success status
        
    Raises:
        HTTPException: If memory deletion fails or not found
    """
    try:
        db = get_db_manager()
        success = await db.delete_memory(current_user.username, memory_id)
        
        if success:
            logger.info(f"Deleted memory {memory_id} for user {current_user.username}")
            return {"success": True}
        
        return {"success": False}
    except DatabaseError as e:
        logger.error(f"Database error during memory deletion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting memory: {str(e)}"
        )


@router.post("/cleanup-memories")
async def cleanup_old_memories(
    days_threshold: int = 30,
    current_user: User = Depends(validate_request_auth)
) -> Dict[str, int]:
    """
    Clean up old memories to prevent Firestore duplication.
    
    Args:
        days_threshold: Age threshold in days for memories to delete
        current_user: Current authenticated user from dependency
        
    Returns:
        Dict: Response with count of deleted memories
        
    Raises:
        HTTPException: If cleanup fails
    """
    try:
        db = get_db_manager()
        count = await db.cleanup_old_memories(current_user.username, days_threshold)
        
        logger.info(f"Cleaned up {count} old memories for user {current_user.username}")
        return {"deleted_count": count}
    except Exception as e:
        logger.error(f"Error cleaning up memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up memories: {str(e)}"
        )


@router.get("/debug", status_code=200)
async def api_debug():
    """
    Debug endpoint to test API router and show available routes.
    
    Returns:
        Dict: Debug information
    """
    return {
        "status": "ok",
        "routes": [
            {"path": str(route.path), "methods": list(route.methods)} 
            for route in router.routes
        ],
        "timestamp": datetime.utcnow().isoformat(),
        "service": "EVA API",
        "version": get_settings().APP_VERSION
    }


@router.get("/test", status_code=200)
async def test_endpoint():
    """
    Simple test endpoint that doesn't require auth.
    
    Returns:
        Dict: Test response
    """
    return {
        "status": "ok",
        "message": "API router is functioning correctly",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/health")
async def api_health_check() -> Dict[str, str]:
    """
    Health check endpoint for API.
    
    Returns:
        Dict: Response with status
    """
    return {"status": "ok", "service": "api"}