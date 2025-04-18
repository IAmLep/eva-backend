"""
Memory Extractor module for EVA backend.

Analyzes conversation text to:
1. Extract potential memories (Core facts, Events, Conversational snippets).
2. Identify explicit memory commands (remember, forget, remind).
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# --- Local Imports ---
from config import get_settings
from llm_service import GeminiService # Use LLM for complex extraction
# Import Memory types/categories for classification
from models import MemorySource, MemoryCategory
from utils import parse_datetime_from_text # Assume a utility for date parsing

# Logger configuration
logger = logging.getLogger(__name__)

# --- Regex for Simple Command Extraction ---
# Basic patterns, can be made more robust
REMEMBER_PATTERN = re.compile(r"(?:remember|recall|note down|don't forget)(?: that)?\s+(.+)", re.IGNORECASE)
FORGET_PATTERN = re.compile(r"(?:forget|delete|remove|clear)(?: that| the memory about)?\s+(.+)", re.IGNORECASE)
REMIND_PATTERN = re.compile(r"remind me(?: to)?\s+(.+?)(?:\s+(?:at|on|in|by|next|tomorrow|tonight)\s+.*|$)", re.IGNORECASE)


# --- Pydantic Models ---
class ExtractedMemory(BaseModel):
    """Represents a potential memory extracted from text."""
    content: str
    source: MemorySource # core, event, conversational
    category: Optional[MemoryCategory] = None # For core memories
    importance: int = Field(default=5, ge=1, le=10)
    entity: Optional[str] = None
    event_time: Optional[datetime] = None
    expiration: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0) # Confidence score

class MemoryCommand(BaseModel):
    """Represents an explicit memory command extracted from text."""
    command_type: str # "remember", "forget", "remind"
    content: Optional[str] = None
    entity: Optional[str] = None
    category: Optional[MemoryCategory] = None # Potential category for remember
    event_time: Optional[datetime] = None # For remind
    expiration: Optional[datetime] = None # For remind


class MemoryExtractor:
    """
    Analyzes text to extract memories and commands using rules and LLM.
    """

    def __init__(self):
        """Initialize memory extractor with settings and LLM service."""
        self.settings = get_settings()
        self.gemini_service = GeminiService() # Initialize LLM service

    async def extract_memory_command(self, text: str) -> Optional[MemoryCommand]:
        """
        Identifies explicit memory commands using regex first.
        (LLM could be used for more complex command understanding).
        """
        # 1. Check for "remember" command
        remember_match = REMEMBER_PATTERN.search(text)
        if remember_match:
            content = remember_match.group(1).strip().rstrip('.?!')
            # Basic category/entity detection (very simple)
            category = MemoryCategory.FACT # Default
            entity = None
            if "my name is" in content.lower(): category = MemoryCategory.PERSONAL_INFO
            elif "i live in" in content.lower(): category = MemoryCategory.PERSONAL_INFO
            elif "i like" in content.lower(): category = MemoryCategory.PREFERENCE
            # TODO: Use NER or LLM prompt for better entity/category extraction

            logger.debug(f"Extracted command: remember '{content}'")
            return MemoryCommand(command_type="remember", content=content, category=category, entity=entity)

        # 2. Check for "forget" command
        forget_match = FORGET_PATTERN.search(text)
        if forget_match:
            content_to_forget = forget_match.group(1).strip().rstrip('.?!')
            # TODO: Extract entity or keywords to help identify the memory
            logger.debug(f"Extracted command: forget '{content_to_forget}'")
            return MemoryCommand(command_type="forget", content=content_to_forget)

        # 3. Check for "remind" command
        remind_match = REMIND_PATTERN.search(text)
        if remind_match:
            reminder_content = remind_match.group(1).strip().rstrip('.?!')
            # Extract time using the utility function
            event_time = await parse_datetime_from_text(text) # Pass full text for context
            if event_time:
                 logger.debug(f"Extracted command: remind '{reminder_content}' at {event_time}")
                 # Default expiration: event_time + 1 day
                 expiration = event_time + timedelta(days=1)
                 return MemoryCommand(
                     command_type="remind",
                     content=reminder_content,
                     event_time=event_time,
                     expiration=expiration
                 )
            else:
                 # If time extraction failed, maybe LLM could clarify or just return partial command
                 logger.warning(f"Remind command detected but failed to parse time from: '{text}'")
                 # Return partial command - handler needs to ask for clarification
                 return MemoryCommand(command_type="remind", content=reminder_content, event_time=None)

        return None # No explicit command found

    async def extract_memories_llm(self, conversation_turn: str) -> List[ExtractedMemory]:
        """
        Uses the LLM to analyze a conversation turn and extract potential memories.
        Focuses on Core facts, Preferences, and potentially significant Events mentioned.
        """
        if not self.settings.FEATURES.get("memory_system") or not self.settings.FEATURES.get("conversation_analysis"):
            return [] # Feature disabled

        # Avoid extraction if a command was likely present
        if await self.extract_memory_command(conversation_turn):
             return []

        memories = []
        try:
            # --- Construct LLM Prompt ---
            # More sophisticated prompt engineering needed here.
            prompt = f"""Analyze the following user statement from a conversation. Identify and extract any potential long-term memories (Core Memories) or significant short-term memories (Events) mentioned.

User Statement:
"{conversation_turn}"

Instructions:
1.  Identify statements that represent core facts about the user, their preferences, relationships, or important details they might want recalled later. Classify these as 'CORE'. Determine a relevant category (PERSONAL_INFO, PREFERENCE, FACT, GOAL, RELATIONSHIP).
2.  Identify statements mentioning specific upcoming events, appointments, deadlines, or tasks with a potential time component. Classify these as 'EVENT'. Attempt to extract the date/time if possible.
3.  For each extracted memory, provide:
    *   `content`: The core information distilled into a concise statement.
    *   `source`: 'CORE' or 'EVENT'.
    *   `category`: (Only for CORE) One of: PERSONAL_INFO, PREFERENCE, FACT, GOAL, RELATIONSHIP.
    *   `importance`: A score from 1 (minor) to 10 (critical). Default 5.
    *   `event_time`: (Only for EVENT) The specific date and time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS) if identifiable, otherwise null.
    *   `entity`: (Optional) The main person, place, or thing the memory is about.
    *   `confidence`: Your confidence (0.0-1.0) in this extraction.
4.  If no significant memories are found, return an empty list.
5.  Format the output as a JSON list of objects, like this:
    ```json
    [
      {{
        "content": "User lives in London.",
        "source": "CORE",
        "category": "PERSONAL_INFO",
        "importance": 7,
        "entity": "London",
        "confidence": 0.9
      }},
      {{
        "content": "Dentist appointment next Tuesday at 3 PM.",
        "source": "EVENT",
        "importance": 8,
        "event_time": "YYYY-MM-DDTHH:MM:SS", // Calculate actual date/time
        "entity": "Dentist appointment",
        "confidence": 0.85
      }}
    ]
    ```

Extracted Memories (JSON list):
"""
            logger.debug(f"Sending text to LLM for memory extraction: '{conversation_turn[:100]}...'")

            # Call LLM (non-streaming)
            response_text, token_info, _ = await self.gemini_service.generate_text(
                prompt,
                temperature=0.2, # Low temp for factual extraction
                max_tokens=512 # Adjust based on expected output size
            )

            # --- Parse LLM Response ---
            # Find JSON block in the response
            json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text, re.MULTILINE)
            if not json_match:
                 # Try parsing directly if no markdown block found
                 try:
                     extracted_data = json.loads(response_text)
                 except json.JSONDecodeError:
                     logger.warning(f"Memory extraction LLM response did not contain valid JSON: {response_text}")
                     return []
            else:
                 try:
                     extracted_data = json.loads(json_match.group(1))
                 except json.JSONDecodeError as e:
                     logger.error(f"Failed to parse JSON from LLM memory extraction response: {e}\nResponse part: {json_match.group(1)}")
                     return []

            if not isinstance(extracted_data, list):
                logger.warning(f"Memory extraction LLM response was not a JSON list: {extracted_data}")
                return []

            # --- Validate and Convert to ExtractedMemory objects ---
            for item in extracted_data:
                 if not isinstance(item, dict): continue
                 try:
                     # Validate source
                     source_val = item.get("source")
                     if source_val == "CORE":
                          source = MemorySource.CORE
                          # Validate category for CORE
                          category_val = item.get("category")
                          try:
                              category = MemoryCategory(category_val) if category_val else MemoryCategory.FACT
                          except ValueError:
                              logger.warning(f"Invalid memory category '{category_val}' from LLM, defaulting to FACT.")
                              category = MemoryCategory.FACT
                     elif source_val == "EVENT":
                          source = MemorySource.EVENT
                          category = None # Events don't have categories in this model
                     else:
                          logger.warning(f"Invalid memory source '{source_val}' from LLM. Skipping.")
                          continue

                     # Parse event_time if present
                     event_time = None
                     event_time_str = item.get("event_time")
                     if event_time_str:
                          try:
                               event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00'))
                          except ValueError:
                               logger.warning(f"LLM provided invalid ISO 8601 event_time: {event_time_str}")
                               # Maybe try parsing with dateutil here?
                               # For now, set to None if invalid format

                     # Create ExtractedMemory object (Pydantic validates types)
                     memory = ExtractedMemory(
                          content=item.get("content", "").strip(),
                          source=source,
                          category=category,
                          importance=int(item.get("importance", 5)),
                          entity=item.get("entity"),
                          event_time=event_time,
                          confidence=float(item.get("confidence", 0.5)),
                          # metadata can be added here if LLM provides more details
                     )

                     # --- Filter based on confidence and importance ---
                     min_importance = self.settings.CORE_MEMORY_IMPORTANCE_THRESHOLD # Use setting
                     min_confidence = 0.6 # Set a minimum confidence threshold

                     if memory.content and memory.importance >= min_importance and memory.confidence >= min_confidence:
                         memories.append(memory)
                         logger.info(f"LLM extracted potential memory (Source: {source.value}, "
                                     f"Importance: {memory.importance}, Confidence: {memory.confidence:.2f}): "
                                     f"'{memory.content[:60]}...'")
                     else:
                          logger.debug(f"LLM extraction skipped (low importance/confidence): {memory.content}")


                 except (KeyError, ValueError, TypeError) as e:
                      logger.warning(f"Failed to parse memory item from LLM response: {item}. Error: {e}")

        except Exception as e:
            logger.error(f"Error during LLM memory extraction: {e}", exc_info=True)

        return memories


# --- Singleton Instance ---
_memory_extractor: Optional[MemoryExtractor] = None

def get_memory_extractor() -> MemoryExtractor:
    """Gets the singleton MemoryExtractor instance."""
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = MemoryExtractor()
    return _memory_extractor


# --- Helper Function (if needed, or move to utils.py) ---
# Placeholder for date parsing utility
async def parse_datetime_from_text(text: str) -> Optional[datetime]:
    """Placeholder: Parses a datetime from text using rules or LLM."""
    # TODO: Implement robust date/time parsing (e.g., using dateutil, duckling, or an LLM call)
    logger.warning("parse_datetime_from_text is a placeholder and needs implementation.")
    # Simple example: Check for "tomorrow"
    if "tomorrow" in text.lower():
         return datetime.now(auth.timezone.utc).replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
    # Add more rules...
    return None # Return None if no date/time found