"""
Memory Extractor module for EVA backend.

Analyzes conversation text to:
1. Extract potential memories (Core facts, Events, Conversational snippets).
2. Identify explicit memory commands (remember, forget, remind).
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

# --- Local Imports ---
from config import settings
from llm_service import GeminiService
from models import MemorySource, MemoryCategory
from utils import parse_datetime_from_text

# Logger configuration
logger = logging.getLogger(__name__)

# --- Regex for Simple Command Extraction ---
REMEMBER_PATTERN = re.compile(r"(?:remember|recall|note down|don't forget)(?: that)?\s+(.+)", re.IGNORECASE)
FORGET_PATTERN = re.compile(r"(?:forget|delete|remove|clear)(?: that| the memory about)?\s+(.+)", re.IGNORECASE)
REMIND_PATTERN = re.compile(
    r"remind me(?: to)?\s+(.+?)(?:\s+(?:at|on|in|by|next|tomorrow|tonight|later|today)\s+.*|$)",
    re.IGNORECASE,
)

# --- Pydantic Models ---
class ExtractedMemory(BaseModel):
    content: str
    source: MemorySource
    category: Optional[MemoryCategory] = None
    importance: int = Field(default=5, ge=1, le=10)
    entity: Optional[str] = None
    event_time: Optional[datetime] = None
    expiration: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class MemoryCommand(BaseModel):
    command_type: str
    content: Optional[str] = None
    entity: Optional[str] = None
    category: Optional[MemoryCategory] = None
    event_time: Optional[datetime] = None
    expiration: Optional[datetime] = None


class MemoryExtractor:
    def __init__(self):
        self.settings = settings
        self.gemini_service = GeminiService()

    async def extract_memory_command(self, text: str) -> Optional[MemoryCommand]:
        # (Implementation unchanged)
        m = REMEMBER_PATTERN.search(text)
        if m:
            content = m.group(1).strip().rstrip('.?!')
            category = MemoryCategory.FACT
            if "my name is" in content.lower() or "i live in" in content.lower():
                category = MemoryCategory.PERSONAL_INFO
            elif "i like" in content.lower():
                category = MemoryCategory.PREFERENCE
            logger.debug(f"Extracted command: remember '{content}'")
            return MemoryCommand(command_type="remember", content=content, category=category)

        m = FORGET_PATTERN.search(text)
        if m:
            content = m.group(1).strip().rstrip('.?!')
            logger.debug(f"Extracted command: forget '{content}'")
            return MemoryCommand(command_type="forget", content=content)

        m = REMIND_PATTERN.search(text)
        if m:
            content = m.group(1).strip().rstrip('.?!')
            event_time = await parse_datetime_from_text(text)
            if event_time:
                expiration = event_time + timedelta(days=1)
                logger.debug(f"Extracted command: remind '{content}' at {event_time}")
                return MemoryCommand(
                    command_type="remind",
                    content=content,
                    event_time=event_time,
                    expiration=expiration,
                )
            else:
                logger.warning(f"Remind command found, but failed to parse time: '{text}'")
                return MemoryCommand(command_type="remind", content=content)

        return None

    async def identify_potential_memory(self, conversation_turn: str) -> Optional[Dict]:
        if await self.extract_memory_command(conversation_turn):
            return None

        try:
            # --- Construct LLM Prompt (Line by Line - NO Triple Quotes) ---
            prompt_lines = [
                "Analyze the following user statement from a conversation. Identify if it contains a single, distinct piece of information that should be stored as a long-term Core Memory (fact, preference, personal detail) or a short-term Event Memory (appointment, reminder with a time).",
                "",
                "User Statement:",
                '"{conversation_turn}"', # Placeholder for format
                "",
                "Instructions:",
                "1.  If the statement contains a core fact, preference, personal detail, goal, or relationship info, identify it. Classify its category (PERSONAL_INFO, PREFERENCE, FACT, GOAL, RELATIONSHIP) and estimate its importance (1-10).",
                "2.  If the statement mentions a specific upcoming event, appointment, deadline, or task with a potential time component, identify it. Mark it as an EVENT. Attempt to extract the date/time if possible (YYYY-MM-DDTHH:MM:SS format).",
                "3.  Focus on extracting only ONE key memory if present. If multiple pieces of info exist, pick the most significant one.",
                '4.  If no significant memory is found, respond with "None".',
                "5.  If a memory is found, respond ONLY with a JSON object containing:",
                "    *   `content`: The core information distilled into a concise statement.",
                '    *   `source`: "CORE" or "EVENT".',
                "    *   `category`: (Only for CORE) One of the categories listed above.",
                "    *   `importance`: (1-10). Default 5.",
                "    *   `event_time`: (Only for EVENT) ISO 8601 string or null.",
                "    *   `entity`: (Optional) The main subject.",
                "    *   `confidence`: Your confidence (0.0-1.0).",
                "",
                "Example Response (Core Memory):",
                "```json",
                "{",
                '  "content": "User\'s favorite color is blue.",', # Escaped single quote
                '  "source": "CORE",',
                '  "category": "PREFERENCE",',
                '  "importance": 6,',
                '  "entity": "favorite color",',
                '  "confidence": 0.9',
                "}",
                "```",
                "Example Response (Event Memory):",
                "```json",
                "{",
                '  "content": "Team meeting scheduled for Friday at 2 PM.",',
                '  "source": "EVENT",',
                '  "importance": 7,',
                '  "event_time": "2025-10-27T14:00:00",',
                '  "entity": "Team meeting",',
                '  "confidence": 0.85',
                "}",
                "```",
                "Example Response (No Memory):",
                "None",
                "",
                'Your Response (JSON object or "None"):',
            ]
            # Join lines and format the conversation turn in
            prompt = "\n".join(prompt_lines).format(conversation_turn=conversation_turn)

            logger.debug(f"Sending text to LLM for potential memory identification: '{conversation_turn[:100]}...'")

            # --- Call LLM Service ---
            response_text, token_info, _ = await self.gemini_service.generate_text(
                prompt,
                temperature=0.2,
                max_tokens=256
            )

            # --- Parse LLM Response ---
            response_text = response_text.strip()
            if response_text.lower() == "none":
                 logger.info("LLM indicated no potential memory found.")
                 return None

            json_match = re.search(r"```json\s*([\s\S]+?)\s*```", response_text, re.MULTILINE)
            json_str = json_match.group(1) if json_match else response_text

            try:
                extracted_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Potential memory LLM response was not 'None' but failed JSON parsing: {e}\nResponse: {response_text}")
                return None

            if not isinstance(extracted_data, dict):
                logger.warning(f"Potential memory LLM response was not a JSON object: {extracted_data}")
                return None

            # --- Validate and Return ---
            if not extracted_data.get("content") or not extracted_data.get("source"):
                 logger.warning(f"Potential memory extraction missing content or source: {extracted_data}")
                 return None

            try:
                 source_val = extracted_data.get("source")
                 extracted_data["source"] = MemorySource(source_val) if source_val else None
                 category_val = extracted_data.get("category")
                 if extracted_data["source"] == MemorySource.CORE:
                     extracted_data["category"] = MemoryCategory(category_val) if category_val else MemoryCategory.FACT
                 else:
                     extracted_data["category"] = None
            except ValueError as e:
                 logger.warning(f"Invalid source or category from LLM: {e}. Data: {extracted_data}")
                 return None

            min_confidence = 0.65
            if extracted_data.get("confidence", 0.0) < min_confidence:
                 logger.info(f"Potential memory skipped due to low confidence ({extracted_data.get('confidence')}): {extracted_data.get('content')}")
                 return None

            logger.info(f"LLM identified potential memory: {extracted_data}")
            return extracted_data

        except Exception as e:
            logger.error(f"Error during LLM potential memory identification: {e}", exc_info=True)
            return None


# --- Singleton Instance ---
_memory_extractor: Optional[MemoryExtractor] = None

def get_memory_extractor() -> MemoryExtractor:
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = MemoryExtractor()
    return _memory_extractor