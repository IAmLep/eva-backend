"""
Context Window Manager for EVA backend.

Manages the conversation context for the LLM, including history,
memory injection, system prompts, token counting, and summarization.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

# --- Local Imports ---
from config import get_settings
# Import specific types from memory_manager
from memory_manager import get_memory_manager, MemoryType, MemoryCategory
from models import User, Memory, Conversation # Import relevant models
from llm_service import GeminiService # Import for summarization

# Logger configuration
logger = logging.getLogger(__name__)

# --- Constants ---
# Rough estimate, adjust based on tokenizer if known
CHARS_PER_TOKEN_ESTIMATE = 4
# Importance scores (relative)
IMPORTANCE_SYSTEM = 3.0
IMPORTANCE_MEMORY_HIGH = 2.5
IMPORTANCE_MEMORY_MEDIUM = 2.0
IMPORTANCE_MEMORY_LOW = 1.5
IMPORTANCE_SUMMARY = 1.8
IMPORTANCE_RECENT_MESSAGE = 1.0
IMPORTANCE_OLD_MESSAGE = 0.5


class ContextItem(BaseModel):
    """Represents an item within the context window."""
    content: str
    token_count: int
    importance: float
    source: str  # e.g., "system", "memory:core", "summary", "message"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationSummary(BaseModel):
    """Represents a summary of part of the conversation."""
    summary_text: str
    turn_count: int # Number of original turns this summary replaces
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None # Placeholder for future sentiment analysis
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ContextWindow:
    """
    Manages the dynamic context window for LLM interactions.

    Handles token limits, context assembly from various sources
    (messages, memories, summaries, system prompts), and triggers
    summarization when needed.
    """

    def __init__(self):
        """Initialize context window manager using settings."""
        self.settings = get_settings()
        self.max_tokens = self.settings.CONTEXT_MAX_TOKENS
        self.summarize_after_turns = self.settings.SUMMARIZE_AFTER_TURNS
        self.keep_recent_messages = 5 # Always keep at least N recent messages

        self.memory_manager = get_memory_manager()
        self.gemini_service = GeminiService() # For summarization

        # --- Context Components ---
        self.system_instructions: List[ContextItem] = []
        self.active_memories: List[ContextItem] = []
        self.summaries: List[ContextItem] = []
        self.recent_messages: List[ContextItem] = []

        # --- State ---
        self.current_token_count: int = 0
        self.current_turn_count: int = 0 # Turns since last summary
        self.mentioned_entities: Dict[str, float] = {} # entity -> importance/recency score

        logger.info(f"Context window initialized: max_tokens={self.max_tokens}, "
                   f"summarize_after={self.summarize_after_turns} turns")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (simple approximation)."""
        if not text:
            return 0
        # A slightly more robust estimate than just chars/4
        # Count words and add a buffer. Real tokenizer is better.
        words = len(text.split())
        chars = len(text)
        return max(words, chars // CHARS_PER_TOKEN_ESTIMATE) + 1

    def _add_item(self, item: ContextItem):
        """Internal method to add an item and update token count."""
        self.current_token_count += item.token_count
        # Note: Specific lists (system, memory, etc.) are appended to directly

    def _remove_item(self, item: ContextItem):
        """Internal method to remove an item and update token count."""
        self.current_token_count -= item.token_count
        # Note: Specific lists must be managed externally

    def add_system_instruction(self, instruction: str, importance: float = IMPORTANCE_SYSTEM):
        """Adds a system instruction."""
        if not instruction: return
        token_count = self._estimate_tokens(instruction)
        item = ContextItem(
            content=f"System Instruction: {instruction}", # Prefix for clarity
            token_count=token_count,
            importance=importance,
            source="system"
        )
        self.system_instructions.append(item)
        self._add_item(item)
        self._manage_token_limit() # Ensure limits after adding
        logger.debug(f"Added system instruction ({token_count} tokens). Total: {self.current_token_count}")

    def add_message(self, role: str, content: str):
        """Adds a user or assistant message."""
        if not content: return
        # Basic sanitization or validation could happen here
        token_count = self._estimate_tokens(content)
        item = ContextItem(
            content=f"{role.capitalize()}: {content}", # Format message
            token_count=token_count,
            importance=IMPORTANCE_RECENT_MESSAGE, # New messages are important initially
            source=f"message:{role}"
        )
        self.recent_messages.append(item)
        self._add_item(item)

        # Update turn count only for user messages to trigger summarization correctly
        if role == "user":
            self.current_turn_count += 1

        # Update entity tracking based on message content
        self._extract_entities(content)

        # Check if summarization is needed (after adding message)
        # We'll call summarize externally or based on turn count check later
        # if self.current_turn_count >= self.summarize_after_turns:
        #    asyncio.create_task(self.summarize_conversation()) # Run async

        self._manage_token_limit() # Ensure limits after adding
        logger.debug(f"Added {role} message ({token_count} tokens). Total: {self.current_token_count}")


    def add_memory(self, memory: Memory, relevance_score: float):
        """Adds a relevant memory to the context."""
        if not memory or not memory.content: return

        # Format memory content for context
        if memory.source == MemoryType.CORE.value:
            category = memory.metadata.get('category', 'Fact')
            formatted_content = f"Relevant Memory ({category}): {memory.content}"
            base_importance = IMPORTANCE_MEMORY_MEDIUM
        elif memory.source == MemoryType.EVENT.value:
            event_time_str = memory.metadata.get("event_time", "Unknown time")
            try: # Format time nicely if possible
                event_dt = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                event_time_str = event_dt.strftime("%Y-%m-%d %H:%M")
            except ValueError: pass
            formatted_content = f"Relevant Event ({event_time_str}): {memory.content}"
            base_importance = IMPORTANCE_MEMORY_HIGH # Events often time-sensitive
        elif memory.source == MemoryType.CONVERSATIONAL.value:
             formatted_content = f"Relevant Past Conversation Snippet: {memory.content}"
             base_importance = IMPORTANCE_MEMORY_LOW
        else:
            formatted_content = f"Relevant Info: {memory.content}"
            base_importance = IMPORTANCE_MEMORY_MEDIUM

        token_count = self._estimate_tokens(formatted_content)
        # Scale importance by relevance and base importance
        importance = base_importance * relevance_score

        item = ContextItem(
            content=formatted_content,
            token_count=token_count,
            importance=importance,
            source=f"memory:{memory.source}"
        )
        self.active_memories.append(item)
        self._add_item(item)
        # Don't manage token limit here, do it after refresh is complete

    def _add_summary(self, summary: ConversationSummary):
        """Adds a conversation summary item."""
        item = ContextItem(
            content=f"Summary of previous conversation ({summary.turn_count} turns): {summary.summary_text}",
            token_count=summary.token_count,
            importance=IMPORTANCE_SUMMARY,
            source="summary",
            timestamp=summary.timestamp
        )
        self.summaries.append(item)
        self._add_item(item)
        # Don't manage token limit here, do it after summarization is complete

    def clear(self):
        """Clears the entire context window."""
        self.system_instructions = []
        self.active_memories = []
        self.summaries = []
        self.recent_messages = []
        self.current_token_count = 0
        self.current_turn_count = 0
        self.mentioned_entities = {}
        logger.info("Context window cleared")

    async def refresh_memories(self, user_id: str):
        """Fetches and adds relevant memories based on recent context."""
        # Clear existing memories first
        self._clear_active_memories()

        # Determine query based on recent messages or entities
        if self.recent_messages:
            # Use content of last few messages as query context
            query_context = "\n".join(
                msg.content for msg in self.recent_messages[-self.keep_recent_messages:]
            )
        else:
            query_context = "General context" # Fallback if no messages

        # Get top N entities based on recency/importance
        top_entities = sorted(
            self.mentioned_entities.items(),
            key=lambda item: item[1], # Sort by score
            reverse=True
        )[:5] # Get top 5 entities
        entity_names = [name for name, score in top_entities]

        try:
            relevant_memories = await self.memory_manager.get_relevant_memories(
                user_id=user_id,
                query=query_context,
                entities=entity_names,
                limit=self.settings.MEMORY_REFRESH_BATCH_SIZE # Use setting
            )

            if relevant_memories:
                logger.info(f"Refreshing memories: Found {len(relevant_memories)} relevant memories.")
                for memory, relevance in relevant_memories:
                    self.add_memory(memory, relevance)
                self._manage_token_limit() # Manage limit after adding all new memories
            else:
                 logger.info("Refreshing memories: No relevant memories found.")

        except Exception as e:
            logger.error(f"Failed to refresh memories for user {user_id}: {e}", exc_info=True)

    def _clear_active_memories(self):
        """Removes all active memories and updates token count."""
        if not self.active_memories:
            return
        removed_tokens = sum(item.token_count for item in self.active_memories)
        self.current_token_count -= removed_tokens
        self.active_memories = []
        logger.debug(f"Cleared active memories (removed {removed_tokens} tokens).")


    def assemble_context(self) -> str:
        """Assembles the context string to be sent to the LLM."""

        # Get all context items
        all_items = (
            self.system_instructions +
            self.active_memories +
            self.summaries +
            self.recent_messages
        )

        # Sort items primarily by importance (desc), then by timestamp (asc) for tie-breaking
        # This ensures important items stay, and among equally important, older ones might be dropped first if needed
        # (though pruning logic primarily uses importance)
        sorted_items = sorted(
            all_items,
            key=lambda x: (x.importance, x.timestamp),
            reverse=True # Higher importance first
        )

        # Build context string respecting max_tokens (conservative approach)
        final_context_parts = []
        current_tokens = 0
        added_sources = set()

        # Always include system instructions first if they fit
        system_tokens = sum(item.token_count for item in self.system_instructions)
        if system_tokens <= self.max_tokens:
            final_context_parts.extend(item.content for item in self.system_instructions)
            current_tokens += system_tokens
            added_sources.add("system")

        # Add other items based on sorted importance until token limit is reached
        for item in sorted_items:
             # Skip system instructions if already added
            if item.source == "system" and "system" in added_sources:
                continue

            # Check if adding this item exceeds the limit
            if current_tokens + item.token_count <= self.max_tokens:
                final_context_parts.append(item.content)
                current_tokens += item.token_count
                added_sources.add(item.source)
            else:
                 # If we can't add the item, break (since they are sorted by importance)
                 # Log which important item got cut off
                 logger.debug(f"Context limit reached. Item dropped: source='{item.source}', "
                              f"importance={item.importance:.2f}, tokens={item.token_count}")
                 break # Stop adding items

        # Structure the final context (optional, helps LLM differentiate sections)
        structured_context = ""
        if any(item.source == "system" for item in all_items if item.content in final_context_parts):
             structured_context += "--- System Instructions ---\n"
             structured_context += "\n".join(item.content for item in self.system_instructions if item.content in final_context_parts) + "\n\n"
        if any(item.source.startswith("memory:") for item in all_items if item.content in final_context_parts):
             structured_context += "--- Relevant Information ---\n"
             structured_context += "\n".join(item.content for item in self.active_memories if item.content in final_context_parts) + "\n\n"
        if any(item.source == "summary" for item in all_items if item.content in final_context_parts):
             structured_context += "--- Conversation Summary ---\n"
             structured_context += "\n".join(item.content for item in self.summaries if item.content in final_context_parts) + "\n\n"
        if any(item.source.startswith("message:") for item in all_items if item.content in final_context_parts):
             structured_context += "--- Current Conversation ---\n"
             # Ensure messages are in chronological order in the final output
             final_messages = sorted(
                 [item for item in self.recent_messages if item.content in final_context_parts],
                 key=lambda x: x.timestamp
             )
             structured_context += "\n".join(item.content for item in final_messages) + "\n"


        logger.info(f"Assembled context: {current_tokens} tokens used out of {self.max_tokens} limit.")
        return structured_context.strip()


    def _extract_entities(self, text: str):
        """Rudimentary entity extraction (proper nouns)."""
        # Decay existing entity scores slightly
        for entity in self.mentioned_entities:
            self.mentioned_entities[entity] *= 0.95

        # Simple regex for capitalized words/phrases (adjust as needed)
        # This is basic; a proper NER model would be much better.
        potential_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        for entity in potential_entities:
            # Increase score for mention, reset decay
            self.mentioned_entities[entity] = self.mentioned_entities.get(entity, 0) * 0.95 + 1.0
            logger.debug(f"Extracted/updated entity: '{entity}'")

        # Prune entities with very low scores
        self.mentioned_entities = {e: s for e, s in self.mentioned_entities.items() if s > 0.1}


    async def summarize_conversation(self) -> bool:
        """
        Checks if summarization is needed and performs it.
        Replaces older messages with a summary item.
        Returns True if summarization was performed, False otherwise.
        """
        if self.current_turn_count < self.summarize_after_turns or len(self.recent_messages) <= self.keep_recent_messages:
            return False # Not enough turns or messages to summarize

        messages_to_summarize = self.recent_messages[:-self.keep_recent_messages]
        if not messages_to_summarize:
            return False # Should not happen based on above check, but safety first

        logger.info(f"Attempting to summarize {len(messages_to_summarize)} messages...")

        # --- Prepare prompt for LLM summarization ---
        conversation_text = "\n".join([item.content for item in messages_to_summarize])
        prompt = f"""Summarize the following conversation excerpt concisely, capturing the key points, decisions, and important information exchanged. Focus on details that would be useful to remember later.

Conversation Excerpt:
---
{conversation_text}
---

Summary:"""

        try:
            # Call LLM to generate summary
            summary_text, token_info, _ = await self.gemini_service.generate_text(
                prompt,
                temperature=0.3, # Lower temp for factual summary
                max_tokens=max(150, self.max_tokens // 10) # Limit summary token size
            )

            if not summary_text:
                 logger.warning("Summarization attempt yielded empty text.")
                 return False

            summary_token_count = self._estimate_tokens(summary_text)
            original_token_count = sum(item.token_count for item in messages_to_summarize)

            # Check if summary is actually shorter (worthwhile)
            if summary_token_count >= original_token_count * 0.9: # Only summarize if it saves > 10% tokens
                 logger.info(f"Summarization deemed not efficient: Original {original_token_count} tokens vs Summary {summary_token_count} tokens.")
                 # Reset turn count anyway to avoid constant re-attempts if summarization isn't helping
                 self.current_turn_count = 0
                 return False

            # --- Create and add summary ---
            summary = ConversationSummary(
                summary_text=summary_text.strip(),
                turn_count=len(messages_to_summarize), # How many messages it replaces
                entities=list(self.mentioned_entities.keys()), # Include current entities snapshot
                token_count=summary_token_count,
                timestamp=messages_to_summarize[-1].timestamp # Timestamp of the last message summarized
            )
            self._add_summary(summary)

            # --- Remove summarized messages ---
            num_to_remove = len(messages_to_summarize)
            removed_items = self.recent_messages[:num_to_remove]
            self.recent_messages = self.recent_messages[num_to_remove:] # Keep only the most recent ones
            for item in removed_items:
                self._remove_item(item)

            self.current_turn_count = 0 # Reset turn count after successful summary
            logger.info(f"Summarized {num_to_remove} messages into {summary_token_count} tokens. "
                       f"Saved {original_token_count - summary_token_count} tokens.")

            self._manage_token_limit() # Ensure limits after replacement
            return True

        except Exception as e:
            logger.error(f"Failed to create conversation summary: {e}", exc_info=True)
            # Don't reset turn count on failure, maybe try again later
            return False

    def _manage_token_limit(self):
        """Prunes context items if total tokens exceed max_tokens."""
        if self.current_token_count <= self.max_tokens:
            return # Within limits

        logger.warning(f"Token limit exceeded ({self.current_token_count}/{self.max_tokens}). Pruning context...")

        # Combine potential items to prune (excluding system instructions)
        prunable_items = (
             self.active_memories +
             self.summaries +
             # Only consider older messages for pruning
             self.recent_messages[:-self.keep_recent_messages]
        )

        # Sort by importance (ascending) then timestamp (ascending) - least important / oldest first
        prunable_items.sort(key=lambda x: (x.importance, x.timestamp))

        items_removed_count = 0
        tokens_saved = 0

        while self.current_token_count > self.max_tokens and prunable_items:
            item_to_remove = prunable_items.pop(0) # Get least important item

            # Find and remove it from its original list
            removed = False
            if item_to_remove in self.active_memories:
                self.active_memories.remove(item_to_remove)
                removed = True
            elif item_to_remove in self.summaries:
                self.summaries.remove(item_to_remove)
                removed = True
            elif item_to_remove in self.recent_messages:
                 # Double check it's not one of the ones we must keep
                 if self.recent_messages.index(item_to_remove) < len(self.recent_messages) - self.keep_recent_messages:
                      self.recent_messages.remove(item_to_remove)
                      removed = True

            if removed:
                self._remove_item(item_to_remove)
                items_removed_count += 1
                tokens_saved += item_to_remove.token_count
                logger.debug(f"Pruned item: source='{item_to_remove.source}', "
                             f"importance={item_to_remove.importance:.2f}, tokens={item_to_remove.token_count}")
            else:
                 logger.warning(f"Could not find item to remove during pruning: {item_to_remove.source}")


        if items_removed_count > 0:
             logger.info(f"Pruning complete. Removed {items_removed_count} items, saved {tokens_saved} tokens. "
                        f"New total: {self.current_token_count}/{self.max_tokens}")
        elif self.current_token_count > self.max_tokens:
             logger.error(f"Failed to bring context within token limits after pruning attempt. "
                         f"Current: {self.current_token_count}/{self.max_tokens}. System instructions might be too large.")


# --- Singleton Instance ---
_context_window: Optional[ContextWindow] = None

def get_context_window() -> ContextWindow:
    """Gets the singleton ContextWindow instance."""
    global _context_window
    if _context_window is None:
        _context_window = ContextWindow()
    return _context_window