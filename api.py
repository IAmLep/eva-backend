"""
API module for EVA backend.

This module provides the main API endpoints for text-based communication
and function calling integration with Gemini API.

Last updated: 2025-04-01
Version: v1.8.6
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from auth import get_current_active_user
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
        user_id: ID of the user sending the message
        context_id: Optional context/conversation ID
        include_memory: Whether to include user memory in context
    """
    text: str
    user_id: str
    context_id: Optional[str] = None
    include_memory: bool = True


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


class MemoryRequest(BaseModel):
    """
    Memory creation request.
    
    Attributes:
        content: Memory content
        metadata: Optional metadata for the memory
    """
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


@router.post("/message", response_model=MessageResponse)
async def process_message(
    message: MessageRequest,
    current_user: User = Depends(get_current_active_user)
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
    # Authorization check
    if current_user.id != message.user_id:
        logger.warning(f"User {current_user.id} attempted to access messages for user {message.user_id}")
        raise AuthorizationError(detail="Not authorized to access this user's messages")
    
    try:
        # Get user memories if requested
        memories = []
        if message.include_memory:
            db = get_db_manager()
            # Get relevant memories that might help with this message
            # In a real implementation, this would use semantic search or other relevance mechanism
            memory_docs = await db.get_relevant_memories(message.user_id, message.text)
            memories = [Memory(**doc) for doc in memory_docs]
            logger.info(f"Retrieved {len(memories)} relevant memories for user {message.user_id}")
        
        # Generate response with LLM
        response_text, context_id, function_calls = await generate_response(
            message.text,
            message.user_id,
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
                    logger.info(f"Executed function call: {call.function.name} for user {message.user_id}")
                    
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


@router.post("/memory", status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory_request: MemoryRequest,
    current_user: User = Depends(get_current_active_user)
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
            user_id=current_user.id,
            content=memory_request.content,
            metadata=memory_request.metadata
        )
        
        memory_id = await db.create_memory(memory)
        logger.info(f"Created memory {memory_id} for user {current_user.id}")
        
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
    current_user: User = Depends(get_current_active_user)
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
        success = await db.delete_memory(current_user.id, memory_id)
        
        if success:
            logger.info(f"Deleted memory {memory_id} for user {current_user.id}")
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
    current_user: User = Depends(get_current_active_user)
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
        count = await db.cleanup_old_memories(current_user.id, days_threshold)
        
        logger.info(f"Cleaned up {count} old memories for user {current_user.id}")
        return {"deleted_count": count}
    except Exception as e:
        logger.error(f"Error cleaning up memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up memories: {str(e)}"
        )


@router.get("/health")
async def api_health_check() -> Dict[str, str]:
    """
    Health check endpoint for API.
    
    Returns:
        Dict: Response with status
    """
    return {"status": "ok", "service": "api"}