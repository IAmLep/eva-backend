"""
Conversation Handler for EVA backend.

This module manages real-time interactions with the Gemini API, including:
- Context window assembly
- Memory refresh and injection
- Streaming response handling

Add this file as a new component to the backend.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from context_window import get_context_window
from memory_manager import get_memory_manager
from llm_service import GeminiService
from models import User

# Logger configuration
logger = logging.getLogger(__name__)

class ConversationHandler:
    """
    Manages real-time conversations and integrates with context and memory systems.
    """
    def __init__(self, user: User):
        """
        Initialize the conversation handler.
        
        Args:
            user: The user initiating the conversation.
        """
        self.user = user
        self.context_window = get_context_window()
        self.memory_manager = get_memory_manager()
        self.gemini_service = GeminiService()
    
    async def process_message(self, message: str) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: The user's input message.
        
        Returns:
            str: The assistant's response.
        """
        try:
            # Add user message to context
            self.context_window.add_message("user", message)
            
            # Refresh memories based on the current message
            await self.context_window.refresh_memories(
                user_id=self.user.user_id,
                current_message=message
            )
            
            # Assemble context for LLM
            context = self.context_window.assemble_context()
            
            # Call Gemini API for a response
            response = await self.gemini_service.stream_conversation(context)
            
            # Add assistant message to context
            self.context_window.add_message("assistant", response)
            
            # Return the response
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return "I'm sorry, something went wrong while processing your message."
    
    async def summarize_conversation(self) -> None:
        """
        Summarize the current conversation to optimize context.
        """
        try:
            await self.context_window._create_conversation_summary()
        except Exception as e:
            logger.error(f"Error summarizing conversation: {e}")