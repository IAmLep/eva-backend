"""
Context Window Manager for EVA backend.

This module provides intelligent management of the conversation context window,
including token tracking, dynamic context assembly, and automatic summarization
to optimize Gemini API interactions.

Current Date: 2025-04-12
Current User: IAmLep
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

from config import get_settings
from memory_manager import get_memory_manager, MemoryType, MemoryRelevance
from models import User, Memory, Conversation

# Logger configuration
logger = logging.getLogger(__name__)


class ContextItem(BaseModel):
    """
    Context item model for context window assembly.
    
    Attributes:
        content: The actual text content
        token_count: Approximate token count
        importance: Importance score (higher = more important)
        source: Source of the context item
        timestamp: When the item was created/added
    """
    content: str
    token_count: int
    importance: float = 1.0
    source: str  # "message", "memory", "summary", etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationSummary(BaseModel):
    """
    Conversation summary model.
    
    Attributes:
        summary_text: Summarized conversation content
        turn_count: Number of conversation turns in the summary
        entities: Key entities mentioned in conversation
        sentiment: Overall sentiment of the conversation
        token_count: Approximate token count
    """
    summary_text: str
    turn_count: int
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    token_count: int


class ContextWindow:
    """
    Dynamic context window manager for optimizing Gemini API interactions.
    
    This class manages the conversation context, handles token limits,
    and provides context assembly for different types of interactions.
    """
    
    def __init__(self, max_tokens: int = 8000, summarize_after_turns: int = 10):
        """
        Initialize context window manager.
        
        Args:
            max_tokens: Maximum tokens in context window
            summarize_after_turns: Number of turns before summarization
        """
        self.settings = get_settings()
        self.max_tokens = max_tokens
        self.summarize_after_turns = summarize_after_turns
        self.memory_manager = get_memory_manager()
        
        # Context components
        self.recent_messages: List[ContextItem] = []
        self.summaries: List[ConversationSummary] = []
        self.active_memories: List[ContextItem] = []
        self.system_instructions: List[ContextItem] = []
        self.current_token_count: int = 0
        self.current_turn_count: int = 0
        
        # Entity tracking
        self.mentioned_entities: Dict[str, float] = {}  # entity -> importance
        
        logger.info(f"Context window initialized: max_tokens={max_tokens}, "
                   f"summarize_after={summarize_after_turns} turns")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a text.
        Simple approximation: ~4 chars per token.
        
        Args:
            text: Text to estimate
            
        Returns:
            int: Estimated token count
        """
        if not text:
            return 0
        return len(text) // 4 + 1  # Simple approximation
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the context window.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        token_count = self.estimate_tokens(content)
        
        # Create context item
        item = ContextItem(
            content=f"{role}: {content}",
            token_count=token_count,
            importance=1.0,  # Recent messages are important
            source="message"
        )
        
        # Add to recent messages
        self.recent_messages.append(item)
        self.current_token_count += token_count
        self.current_turn_count += 1
        
        # Extract entities from message
        self._extract_entities(content)
        
        # Check if summarization is needed
        if self.current_turn_count >= self.summarize_after_turns:
            self._create_conversation_summary()
        
        # Ensure we're within token limits
        self._manage_token_limit()
        
        logger.debug(f"Added {role} message: {token_count} tokens, "
                    f"total: {self.current_token_count}/{self.max_tokens} tokens")
    
    def add_system_instruction(self, instruction: str, importance: float = 2.0) -> None:
        """
        Add a system instruction to the context window.
        
        Args:
            instruction: System instruction text
            importance: Importance score (higher = harder to drop)
        """
        token_count = self.estimate_tokens(instruction)
        
        # Create context item
        item = ContextItem(
            content=f"System: {instruction}",
            token_count=token_count,
            importance=importance,
            source="system"
        )
        
        self.system_instructions.append(item)
        self.current_token_count += token_count
        
        # Ensure we're within token limits
        self._manage_token_limit()
        
        logger.debug(f"Added system instruction: {token_count} tokens")
    
    def add_memory(self, memory: Memory, relevance_score: float) -> None:
        """
        Add a memory to the context window.
        
        Args:
            memory: Memory object to add
            relevance_score: Relevance to current conversation (0-1)
        """
        # Format memory based on type
        if memory.source == "core":
            formatted_content = f"Memory ({memory.metadata.get('category', 'fact')}): {memory.content}"
        elif memory.source == "event":
            event_time = memory.metadata.get("event_time", "")
            formatted_content = f"Upcoming event ({event_time}): {memory.content}"
        else:
            formatted_content = f"Past conversation: {memory.content}"
        
        token_count = self.estimate_tokens(formatted_content)
        
        # Create context item
        item = ContextItem(
            content=formatted_content,
            token_count=token_count,
            importance=relevance_score * 2.0,  # Scale importance by relevance
            source=f"memory:{memory.source}"
        )
        
        self.active_memories.append(item)
        self.current_token_count += token_count
        
        # Ensure we're within token limits
        self._manage_token_limit()
        
        logger.debug(f"Added memory: {token_count} tokens, relevance: {relevance_score:.2f}")
    
    def clear(self) -> None:
        """Clear the context window completely."""
        self.recent_messages = []
        self.summaries = []
        self.active_memories = []
        self.system_instructions = []
        self.current_token_count = 0
        self.current_turn_count = 0
        self.mentioned_entities = {}
        
        logger.info("Context window cleared")
    
    async def refresh_memories(self, user_id: str, current_message: str) -> None:
        """
        Refresh relevant memories based on current conversation.
        
        Args:
            user_id: User ID to fetch memories for
            current_message: Current message for relevance
        """
        # Get top entities
        top_entities = [entity for entity, _ in sorted(
            self.mentioned_entities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]]  # Get top 5 entities
        
        # Fetch relevant memories
        relevant_memories = await self.memory_manager.get_relevant_memories(
            user_id=user_id,
            query=current_message,
            entities=top_entities,
            limit=5
        )
        
        # Clear current memories
        old_memories = self.active_memories
        self.active_memories = []
        self.current_token_count -= sum(m.token_count for m in old_memories)
        
        # Add new memories
        for memory, relevance in relevant_memories:
            self.add_memory(memory, relevance)
            
        logger.info(f"Refreshed memories: added {len(relevant_memories)} relevant memories")
    
    def assemble_context(self) -> str:
        """
        Assemble the full context for the LLM.
        
        Returns:
            str: Assembled context
        """
        # Start with system instructions (highest priority)
        context_parts = [item.content for item in sorted(
            self.system_instructions, 
            key=lambda x: x.importance, 
            reverse=True
        )]
        
        # Add memories (second priority)
        if self.active_memories:
            context_parts.append("Relevant memories:")
            memory_parts = [item.content for item in sorted(
                self.active_memories, 
                key=lambda x: x.importance, 
                reverse=True
            )]
            context_parts.extend(memory_parts)
        
        # Add most recent summary if we have one
        if self.summaries:
            most_recent_summary = self.summaries[-1]
            context_parts.append(f"Previous conversation summary: {most_recent_summary.summary_text}")
        
        # Add recent messages (always include)
        if self.recent_messages:
            context_parts.append("Recent conversation:")
            message_parts = [item.content for item in self.recent_messages]
            context_parts.extend(message_parts)
        
        # Assemble final context
        return "\n\n".join(context_parts)
    
    def get_mentioned_entities(self) -> List[str]:
        """
        Get list of entities mentioned in conversation.
        
        Returns:
            List[str]: List of entity names
        """
        return list(self.mentioned_entities.keys())
    
    def _extract_entities(self, text: str) -> None:
        """
        Extract potential entities from text.
        This is a simple implementation - in production you might
        use NLP for better entity extraction.
        
        Args:
            text: Text to extract entities from
        """
        # Simple entity extraction - proper nouns and capitalized phrases
        # This is a naive implementation - in production use a proper NER model
        potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        # Update entity importance
        for entity in potential_entities:
            if entity in self.mentioned_entities:
                # Increase importance for repeated mentions
                self.mentioned_entities[entity] += 0.5
            else:
                # New entity
                self.mentioned_entities[entity] = 1.0
    
    async def _create_conversation_summary(self) -> None:
        """Create a summary of the current conversation."""
        if not self.recent_messages:
            return
            
        # Collect message contents
        message_texts = [item.content for item in self.recent_messages]
        messages_to_summarize = "\n".join(message_texts)
        
        # In a real implementation, we'd use the LLM to create this summary
        # For now, we'll create a placeholder
        summary_text = f"Conversation about {', '.join(list(self.mentioned_entities.keys())[:3])}"
        summary_tokens = self.estimate_tokens(summary_text)
        
        # Create summary object
        summary = ConversationSummary(
            summary_text=summary_text,
            turn_count=self.current_turn_count,
            entities=list(self.mentioned_entities.keys()),
            token_count=summary_tokens
        )
        
        # Add summary and reset turns
        self.summaries.append(summary)
        self.current_turn_count = 0
        
        # In a real implementation, you might decide to clear some messages after summarization
        # But this is a simple approach that keeps all messages in memory until token limit is hit
        logger.info(f"Created conversation summary: {summary_tokens} tokens")
    
    def _manage_token_limit(self) -> None:
        """
        Ensure context stays within token limit by removing low-importance items.
        """
        # If we're under the limit, no action needed
        if self.current_token_count <= self.max_tokens:
            return
            
        # First, try removing lowest importance memories
        if self.active_memories:
            sorted_memories = sorted(self.active_memories, key=lambda x: x.importance)
            while (self.current_token_count > self.max_tokens and sorted_memories):
                item = sorted_memories.pop(0)  # Remove lowest importance memory
                self.active_memories.remove(item)
                self.current_token_count -= item.token_count
                logger.debug(f"Removed memory to meet token limit: {item.content[:30]}...")
        
        # Next, try removing older messages if we're still over the limit
        if self.current_token_count > self.max_tokens and len(self.recent_messages) > 5:
            # Always keep at least the 5 most recent messages
            # Remove older messages first
            while (self.current_token_count > self.max_tokens and len(self.recent_messages) > 5):
                item = self.recent_messages.pop(0)  # Remove oldest message
                self.current_token_count -= item.token_count
                logger.debug(f"Removed old message to meet token limit: {item.content[:30]}...")
        
        # If we're still over limit after these steps, this is a more complex scenario
        # In a production system, you might implement more sophisticated pruning strategies
        if self.current_token_count > self.max_tokens:
            logger.warning(f"Context still over token limit after pruning: "
                          f"{self.current_token_count}/{self.max_tokens}")


# Singleton instance
_context_window: Optional[ContextWindow] = None


def get_context_window() -> ContextWindow:
    """
    Get the context window singleton.
    
    Returns:
        ContextWindow: Context window manager instance
    """
    global _context_window
    if _context_window is None:
        settings = get_settings()
        max_tokens = getattr(settings, "CONTEXT_MAX_TOKENS", 8000)
        summarize_after = getattr(settings, "SUMMARIZE_AFTER_TURNS", 10)
        _context_window = ContextWindow(max_tokens, summarize_after)
    return _context_window