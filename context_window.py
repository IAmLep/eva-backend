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
from config import settings
from memory_manager import get_memory_manager, MemoryType
from models import Memory
from llm_service import GeminiService

# Logger configuration
logger = logging.getLogger(__name__)

# --- Constants ---
CHARS_PER_TOKEN_ESTIMATE = 4
IMPORTANCE_SYSTEM = 3.0
IMPORTANCE_MEMORY_HIGH = 2.5
IMPORTANCE_MEMORY_MEDIUM = 2.0
IMPORTANCE_MEMORY_LOW = 1.5
IMPORTANCE_SUMMARY = 1.8
IMPORTANCE_RECENT_MESSAGE = 1.0


class ContextItem(BaseModel):
    content: str
    token_count: int
    importance: float
    source: str  # e.g., "system", "memory:core", "summary", "message:user"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationSummary(BaseModel):
    summary_text: str
    turn_count: int
    entities: List[str] = Field(default_factory=list)
    sentiment: Optional[str] = None
    token_count: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ContextWindow:
    """
    Manages the dynamic context window for LLM interactions.
    """

    def __init__(self):
        """Initialize context window manager using settings."""
        self.settings = settings
        self.max_tokens = self.settings.CONTEXT_MAX_TOKENS
        self.summarize_after_turns = self.settings.SUMMARIZE_AFTER_TURNS
        self.keep_recent_messages = 5

        self.memory_manager = get_memory_manager()
        self.gemini_service = GeminiService()

        # Context lists
        self.system_instructions: List[ContextItem] = []
        self.active_memories: List[ContextItem] = []
        self.summaries: List[ContextItem] = []
        self.recent_messages: List[ContextItem] = []

        # State tracking
        self.current_token_count: int = 0
        self.current_turn_count: int = 0
        self.mentioned_entities: Dict[str, float] = {}

        logger.info(
            f"Context window initialized: max_tokens={self.max_tokens}, "
            f"summarize_after={self.summarize_after_turns} turns"
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (approximate)."""
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        return max(words, chars // CHARS_PER_TOKEN_ESTIMATE) + 1

    def _add_item(self, item: ContextItem):
        """Internal: add a context item."""
        self.current_token_count += item.token_count

    def _remove_item(self, item: ContextItem):
        """Internal: remove a context item."""
        self.current_token_count -= item.token_count

    def add_system_instruction(self, instruction: str, importance: float = IMPORTANCE_SYSTEM):
        """Adds a system instruction to context."""
        if not instruction:
            return
        token_count = self._estimate_tokens(instruction)
        item = ContextItem(
            content=f"System Instruction: {instruction}",
            token_count=token_count,
            importance=importance,
            source="system"
        )
        self.system_instructions.append(item)
        self._add_item(item)
        self._manage_token_limit()
        logger.debug(f"Added system instruction ({token_count} tokens). Total: {self.current_token_count}")

    def add_message(self, role: str, content: str):
        """Adds a user or assistant message to the context window."""
        if not content:
            return
        token_count = self._estimate_tokens(content)
        item = ContextItem(
            content=f"{role.capitalize()}: {content}",
            token_count=token_count,
            importance=IMPORTANCE_RECENT_MESSAGE,
            source=f"message:{role}"
        )
        self.recent_messages.append(item)
        self._add_item(item)

        if role == "user":
            self.current_turn_count += 1

        self._extract_entities(content)
        self._manage_token_limit()
        logger.debug(
            f"Added {role} message ({token_count} tokens). "
            f"Total tokens now: {self.current_token_count}"
        )

    def add_memory(self, memory: Memory, relevance_score: float):
        """Adds a relevant memory to context."""
        if not memory or not memory.content:
            return

        # format content & importance by type
        if memory.source == MemoryType.CORE.value:
            formatted = f"Core Memory: {memory.content}"
            base_imp = IMPORTANCE_MEMORY_MEDIUM
        elif memory.source == MemoryType.EVENT.value:
            formatted = f"Event Memory: {memory.content}"
            base_imp = IMPORTANCE_MEMORY_HIGH
        else:
            formatted = f"Memory: {memory.content}"
            base_imp = IMPORTANCE_MEMORY_LOW

        token_count = self._estimate_tokens(formatted)
        importance = base_imp * relevance_score
        item = ContextItem(
            content=formatted,
            token_count=token_count,
            importance=importance,
            source=f"memory:{memory.source}"
        )
        self.active_memories.append(item)
        self._add_item(item)

    def _add_summary(self, summary: ConversationSummary):
        """Adds a summary item to context."""
        item = ContextItem(
            content=f"Summary: {summary.summary_text}",
            token_count=summary.token_count,
            importance=IMPORTANCE_SUMMARY,
            source="summary",
            timestamp=summary.timestamp
        )
        self.summaries.append(item)
        self._add_item(item)

    def clear(self):
        """Clears the context window completely."""
        self.system_instructions.clear()
        self.active_memories.clear()
        self.summaries.clear()
        self.recent_messages.clear()
        self.current_token_count = 0
        self.current_turn_count = 0
        self.mentioned_entities.clear()
        logger.info("Context window cleared")

    async def refresh_memories(self, user_id: str):
        """Fetches and adds relevant memories."""
        # clear existing
        removed = sum(item.token_count for item in self.active_memories)
        self.current_token_count -= removed
        self.active_memories.clear()

        # prepare query
        context = "\n".join(m.content for m in self.recent_messages[-self.keep_recent_messages:])
        entities = list(sorted(self.mentioned_entities, key=self.mentioned_entities.get, reverse=True)[:5])

        memories = await self.memory_manager.get_relevant_memories(
            user_id=user_id,
            query=context,
            entities=entities,
            limit=self.settings.MEMORY_REFRESH_BATCH_SIZE
        )
        for mem, score in memories:
            self.add_memory(mem, score)
        self._manage_token_limit()

    def assemble_context(self) -> str:
        """Builds the final prompt context string."""
        all_items = (
            self.system_instructions +
            self.active_memories +
            self.summaries +
            self.recent_messages
        )
        # sort by importance desc, timestamp asc
        items = sorted(
            all_items,
            key=lambda x: (x.importance, -x.timestamp.timestamp()),
            reverse=True
        )
        parts = []
        used = 0
        for item in items:
            if used + item.token_count <= self.max_tokens:
                parts.append(item.content)
                used += item.token_count
            else:
                break
        logger.info(f"Assembled context: {used}/{self.max_tokens} tokens")
        return "\n".join(parts)

    def _extract_entities(self, text: str):
        """Simple proper‐noun extraction."""
        for e in list(self.mentioned_entities):
            self.mentioned_entities[e] *= 0.95
        found = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text)
        for ent in found:
            self.mentioned_entities[ent] = self.mentioned_entities.get(ent, 0) + 1.0

    async def summarize_conversation(self) -> bool:
        """Runs summarization when enough turns have passed."""
        if self.current_turn_count < self.summarize_after_turns:
            return False
        to_summarize = self.recent_messages[:-self.keep_recent_messages]
        if not to_summarize:
            return False

        text = "\n".join(m.content for m in to_summarize)
        prompt = f"Summarize:\n{text}"
        summary_text, _, _ = await self.gemini_service.generate_text(
            prompt,
            temperature=0.3,
            max_tokens=min(150, self.max_tokens // 10)
        )
        token_count = self._estimate_tokens(summary_text)
        summary = ConversationSummary(
            summary_text=summary_text.strip(),
            turn_count=len(to_summarize),
            entities=list(self.mentioned_entities),
            token_count=token_count,
            timestamp=to_summarize[-1].timestamp
        )
        self._add_summary(summary)
        for old in to_summarize:
            self._remove_item(old)
        self.recent_messages = self.recent_messages[-self.keep_recent_messages:]
        self.current_turn_count = 0
        self._manage_token_limit()
        return True

    def _manage_token_limit(self):
        """Prune least‐important items if over limit."""
        if self.current_token_count <= self.max_tokens:
            return
        pool = (
            self.active_memories +
            self.summaries +
            self.recent_messages[:-self.keep_recent_messages]
        )
        pool.sort(key=lambda x: (x.importance, x.timestamp))
        while self.current_token_count > self.max_tokens and pool:
            rem = pool.pop(0)
            if rem in self.active_memories:
                self.active_memories.remove(rem)
            elif rem in self.summaries:
                self.summaries.remove(rem)
            elif rem in self.recent_messages:
                self.recent_messages.remove(rem)
            self._remove_item(rem)
        logger.debug(f"After prune: {self.current_token_count}/{self.max_tokens} tokens")


# Singleton accessor
_context_window: Optional[ContextWindow] = None

def get_context_window() -> ContextWindow:
    """Returns the singleton ContextWindow."""
    global _context_window
    if _context_window is None:
        _context_window = ContextWindow()
    return _context_window