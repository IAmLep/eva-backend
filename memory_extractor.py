"""
Memory Extractor for EVA backend.

This module provides functionality for extracting memories from conversations
using both rule-based approaches and Gemini API for deeper understanding.

Replace your existing memory_extractor.py file with this enhanced version.

Current Date: 2025-04-13 11:13:26
Current User: IAmLep
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from models import Memory, MemoryCategory, User
from llm_service import GeminiService
from memory_manager import get_memory_manager, MemoryType, MemoryCommand
from config import get_settings

# Logger configuration
logger = logging.getLogger(__name__)


class MemoryExtractor:
    """
    Memory extractor for identifying and extracting memories from conversations.
    
    This class provides functionality for identifying important information
    in conversations that should be stored as memories, as well as explicit
    memory commands.
    """
    
    def __init__(self):
        """Initialize the memory extractor."""
        self.settings = get_settings()
        self.gemini_service = GeminiService()
        self.memory_manager = get_memory_manager()
        
        # Patterns for rule-based extraction
        self.patterns = {
            "remember_cmd": re.compile(r"^(?:please\s+)?remember\s+(?:that\s+)?(.+)$", re.IGNORECASE),
            "remind_cmd": re.compile(r"^(?:please\s+)?remind\s+(?:me\s+)?(?:to\s+)?(.+?)(?:\s+at\s+|by\s+|on\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)?(\d{4}-\d{2}-\d{2})?", re.IGNORECASE),
            "forget_cmd": re.compile(r"^(?:please\s+)?forget\s+(?:about\s+)?(.+)$", re.IGNORECASE),
            "preferences": re.compile(r"(?:i\s+(?:really\s+)?(?:like|love|enjoy|prefer|hate|dislike|can't\s+stand))\s+(.+)", re.IGNORECASE),
            "personal_fact": re.compile(r"(?:my\s+name\s+is|i\s+am\s+from|i\s+live\s+in|i\s+work\s+at|i\s+work\s+as)\s+(.+)", re.IGNORECASE)
        }
        
        logger.info("Memory extractor initialized")
    
    async def extract_memory_command(self, text: str) -> Optional[MemoryCommand]:
        """
        Extract explicit memory commands from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Optional[MemoryCommand]: Memory command if found
        """
        if not text:
            return None
            
        # Check for remember command
        match = self.patterns["remember_cmd"].match(text)
        if match:
            content = match.group(1).strip()
            
            # Try to determine category and entity
            category = MemoryCategory.FACT  # Default
            entity = None
            
            if "I like" in content or "I love" in content or "I enjoy" in content:
                category = MemoryCategory.PREFERENCE
            
            logger.info(f"Extracted remember command: {content}")
            
            return MemoryCommand(
                command_type="remember",
                content=content,
                entity=entity,
                category=category
            )
        
        # Check for remind command
        match = self.patterns["remind_cmd"].match(text)
        if match:
            content = match.group(1).strip()
            time_str = match.group(2)
            date_str = match.group(3)
            
            # Parse time
            event_time = None
            if time_str:
                try:
                    # Handle various time formats
                    now = datetime.now()
                    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    
                    # Simple parsing
                    if ":" in time_str:
                        # Format like "3:30pm"
                        time_parts = time_str.strip().lower()
                        hour_part, minute_part = time_parts.split(":")
                        
                        hour = int(hour_part)
                        is_pm = "pm" in minute_part.lower()
                        minute_part = minute_part.replace("am", "").replace("pm", "").strip()
                        minute = int(minute_part)
                        
                        if is_pm and hour < 12:
                            hour += 12
                            
                        event_time = today.replace(hour=hour, minute=minute)
                    else:
                        # Format like "3pm"
                        time_parts = time_str.strip().lower()
                        is_pm = "pm" in time_parts
                        hour = int(time_parts.replace("am", "").replace("pm", "").strip())
                        
                        if is_pm and hour < 12:
                            hour += 12
                            
                        event_time = today.replace(hour=hour, minute=0)
                    
                    # If time is in the past today, move to tomorrow
                    if event_time < now:
                        event_time += timedelta(days=1)
                        
                except ValueError:
                    logger.warning(f"Could not parse time from {time_str}")
            
            logger.info(f"Extracted remind command: {content}, time: {event_time}")
            
            return MemoryCommand(
                command_type="remind",
                content=content,
                event_time=event_time
            )
        
        # Check for forget command
        match = self.patterns["forget_cmd"].match(text)
        if match:
            content = match.group(1).strip()
            logger.info(f"Extracted forget command: {content}")
            
            return MemoryCommand(
                command_type="forget",
                content=content
            )
        
        # No explicit command found
        return None
    
    async def identify_potential_memory(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Identify potential memory in text using rule-based approach.
        
        Args:
            text: Text to analyze
            
        Returns:
            Optional[Dict[str, Any]]: Memory parameters if found
        """
        if not text:
            return None
            
        # Check for preferences
        match = self.patterns["preferences"].search(text.lower())
        if match:
            preference = match.group(1).strip()
            logger.info(f"Identified preference: {preference}")
            
            return {
                "content": f"User {preference}",
                "category": MemoryCategory.PREFERENCE,
                "importance": 6
            }
        
        # Check for personal facts
        match = self.patterns["personal_fact"].search(text.lower())
        if match:
            fact = match.group(0).strip()  # Get the whole match
            logger.info(f"Identified personal fact: {fact}")
            
            return {
                "content": fact,
                "category": MemoryCategory.FACT,
                "importance": 8
            }
        
        return None
    
    async def extract_memories_with_gemini(
        self,
        conversation: List[Dict[str, str]],
        user: User
    ) -> List[Dict[str, Any]]:
        """
        Use Gemini API to extract important memories from a conversation.
        
        Args:
            conversation: List of message dictionaries (role, content)
            user: User object
            
        Returns:
            List[Dict[str, Any]]: List of memory parameters
        """
        try:
            # Format conversation for Gemini
            formatted_conversation = ""
            for msg in conversation:
                formatted_conversation += f"{msg['role']}: {msg['content']}\n"
            
            # Create prompt for memory extraction
            prompt = f"""
            You are EVA's memory extraction system. Analyze this conversation and identify important information 
            that should be remembered as long-term memories or tracked as events.
            
            CONVERSATION:
            {formatted_conversation}
            
            Extract memories in the following categories:
            - PERSON: Information about people in the user's life
            - PLACE: Locations that are important to the user
            - PREFERENCE: User likes, dislikes, preferences
            - FACT: Important facts about the user
            - EVENT: Time-based information (appointments, deadlines)
            
            For each memory, include:
            - category: The category of memory (one of the above)
            - content: The exact information to remember
            - importance: A score from 1-10 indicating how important this is to remember
            - entity: The main subject of this memory (if applicable)
            
            Format the output as a JSON array of memory objects.
            Only include genuinely important information that would be valuable to remember.
            If there's nothing important to remember, return an empty array.
            """
            
            # Call Gemini API
            response, _, _ = await self.gemini_service.generate_text(prompt)
            
            # Extract JSON
            try:
                start_index = response.find('[')
                end_index = response.rfind(']') + 1
                
                if start_index >= 0 and end_index > start_index:
                    memories_json = response[start_index:end_index]
                    memories = json.loads(memories_json)
                    
                    logger.info(f"Extracted {len(memories)} memories using Gemini")
                    return memories
                else:
                    logger.warning("Failed to extract valid JSON from Gemini response")
                    return []
            
            except json.JSONDecodeError:
                logger.warning("Failed to parse memory extraction response as JSON")
                return []
            
        except Exception as e:
            logger.error(f"Error extracting memories with Gemini: {str(e)}")
            return []
    
    async def create_memories_from_conversation(
        self,
        user_id: str,
        conversation: List[Dict[str, str]],
        conversation_id: str
    ) -> List[Memory]:
        """
        Process a conversation and create relevant memories.
        
        Args:
            user_id: User ID
            conversation: List of message dictionaries
            conversation_id: Conversation ID
            
        Returns:
            List[Memory]: Created memories
        """
        created_memories = []
        
        try:
            # Filter for user messages only
            user_messages = [msg for msg in conversation if msg["role"] == "user"]
            user = User(id=user_id, username="User", email="user@example.com")
            
            # First check for explicit memory commands
            for msg in user_messages:
                command = await self.extract_memory_command(msg["content"])
                
                if command:
                    if command.command_type == "remember":
                        # Create core memory
                        memory = await self.memory_manager.create_core_memory(
                            user_id=user_id,
                            content=command.content,
                            category=command.category or MemoryCategory.FACT,
                            entity=command.entity,
                            importance=7,
                            metadata={"source_conversation": conversation_id}
                        )
                        created_memories.append(memory)
                        
                    elif command.command_type == "remind" and command.event_time:
                        # Create event memory
                        memory = await self.memory_manager.create_event_memory(
                            user_id=user_id,
                            content=command.content,
                            event_time=command.event_time,
                            metadata={"source_conversation": conversation_id}
                        )
                        created_memories.append(memory)
            
            # Then use Gemini for deeper understanding if enabled
            if self.settings.FEATURES.get("memory_system", True):
                extracted_memories = await self.extract_memories_with_gemini(conversation, user)
                
                for memory_data in extracted_memories:
                    category = memory_data.get("category", "FACT").upper()
                    
                    # Skip if extracted memory doesn't meet minimum importance threshold
                    importance = memory_data.get("importance", 5)
                    if importance < self.settings.CORE_MEMORY_IMPORTANCE_THRESHOLD:
                        continue
                    
                    # Create appropriate memory based on category
                    if category == "EVENT":
                        # For events, would need additional processing for time
                        # Simplified version just creates a core memory
                        memory = await self.memory_manager.create_core_memory(
                            user_id=user_id,
                            content=memory_data.get("content", ""),
                            category=MemoryCategory.FACT,
                            entity=memory_data.get("entity"),
                            importance=importance,
                            metadata={
                                "source_conversation": conversation_id,
                                "auto_extracted": True
                            }
                        )
                    else:
                        # Map category string to enum
                        try:
                            memory_category = MemoryCategory[category]
                        except KeyError:
                            memory_category = MemoryCategory.FACT
                        
                        memory = await self.memory_manager.create_core_memory(
                            user_id=user_id,
                            content=memory_data.get("content", ""),
                            category=memory_category,
                            entity=memory_data.get("entity"),
                            importance=importance,
                            metadata={
                                "source_conversation": conversation_id,
                                "auto_extracted": True
                            }
                        )
                    
                    created_memories.append(memory)
            
            return created_memories
            
        except Exception as e:
            logger.error(f"Error creating memories from conversation: {str(e)}")
            return created_memories


# Singleton instance
_memory_extractor: Optional[MemoryExtractor] = None


def get_memory_extractor() -> MemoryExtractor:
    """
    Get the memory extractor singleton.
    
    Returns:
        MemoryExtractor: Memory extractor instance
    """
    global _memory_extractor
    if _memory_extractor is None:
        _memory_extractor = MemoryExtractor()
    return _memory_extractor