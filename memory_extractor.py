import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from llm_service import llm_service
from firestore_manager import firestore_manager
from logging_config import setup_logging

# Setup logging
logger = logging.getLogger(__name__)

class MemoryExtractor:
    """
    Extract key information from conversations to enhance contextual memory
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Memory Extractor initialized")
    
    async def extract_key_info(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """
        Extract key information from a conversation and save it to Firestore
        
        Args:
            conversation_id: The ID of the conversation
            user_id: The ID of the user who owns the conversation
            
        Returns:
            Dict with extracted memory information
        """
        try:
            # Get the conversation with ownership verification
            conversation = await self._get_conversation_with_verification(conversation_id, user_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found or not owned by user {user_id}")
                return {"error": "Conversation not found or not owned by user"}
            
            # Get existing memory for this conversation
            existing_memory = await self.get_conversation_memory(conversation_id, user_id)
            
            # Prepare messages for the extraction prompt
            messages = conversation.get("messages", [])
            if not messages:
                logger.warning(f"No messages found in conversation {conversation_id}")
                return {"error": "No messages in conversation"}
            
            # Create the extraction prompt
            prompt = self._create_extraction_prompt(messages, existing_memory)
            
            # Call LLM to extract information
            response = await llm_service.generate_text(prompt)
            
            # Parse the response
            memory_data = self._parse_memory_response(response)
            
            # Save the extracted memory to Firestore
            timestamp = datetime.utcnow().isoformat()
            memory_entry = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "extracted_at": timestamp,
                "memory_data": memory_data
            }
            
            await firestore_manager.add_or_update_conversation_memory(
                conversation_id, 
                user_id, 
                memory_entry
            )
            
            logger.info(f"Successfully extracted memory for conversation {conversation_id}")
            return memory_entry
            
        except ValueError as e:
            logger.error(f"Value error in extract_key_info: {str(e)}")
            return {"error": f"Value error: {str(e)}"}
            
        except TypeError as e:
            logger.error(f"Type error in extract_key_info: {str(e)}")
            return {"error": f"Type error: {str(e)}"}
            
        except Exception as e:
            logger.error(f"Unexpected error in extract_key_info: {str(e)}")
            return {"error": f"Extraction failed: {str(e)}"}
    
    async def get_conversation_memory(self, conversation_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all memory entries for a conversation
        
        Args:
            conversation_id: The ID of the conversation
            user_id: The ID of the user who owns the conversation
            
        Returns:
            List of memory entries
        """
        try:
            # Verify conversation ownership
            conversation = await self._get_conversation_with_verification(conversation_id, user_id)
            if not conversation:
                logger.warning(f"Conversation {conversation_id} not found or not owned by user {user_id}")
                return []
            
            # Get memory entries
            memory_entries = await firestore_manager.get_conversation_memory(conversation_id, user_id)
            return memory_entries
            
        except ValueError as e:
            logger.error(f"Value error in get_conversation_memory: {str(e)}")
            return []
            
        except TypeError as e:
            logger.error(f"Type error in get_conversation_memory: {str(e)}")
            return []
            
        except Exception as e:
            logger.error(f"Unexpected error in get_conversation_memory: {str(e)}")
            return []
    
    async def get_all_memories_for_user(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all memory entries for all conversations of a user
        
        Args:
            user_id: The ID of the user
            
        Returns:
            Dict mapping conversation_id to memory entries
        """
        try:
            # Get all conversations for the user
            conversations = await firestore_manager.get_conversations_by_user(user_id)
            
            # Get memory for each conversation
            result = {}
            for conversation in conversations:
                conversation_id = conversation.get("id")
                if conversation_id:
                    memory = await self.get_conversation_memory(conversation_id, user_id)
                    if memory:
                        result[conversation_id] = memory
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting all memories for user {user_id}: {str(e)}")
            return {}
    
    async def _get_conversation_with_verification(
        self, 
        conversation_id: str, 
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Helper method to get conversation with ownership verification"""
        conversation = await firestore_manager.get_conversation(conversation_id)
        
        # Verify ownership
        if not conversation or conversation.get("user_id") != user_id:
            return None
            
        return conversation
    
    def _create_extraction_prompt(
        self, 
        messages: List[Dict[str, Any]], 
        existing_memory: List[Dict[str, Any]]
    ) -> str:
        """Create a prompt for memory extraction"""
        # Format messages
        formatted_messages = "\n".join([
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" 
            for msg in messages
        ])
        
        # Format existing memory if available
        memory_context = ""
        if existing_memory:
            memory_items = []
            for entry in existing_memory:
                if "memory_data" in entry and "key_points" in entry["memory_data"]:
                    memory_items.extend(entry["memory_data"]["key_points"])
            
            if memory_items:
                memory_context = "Previously extracted information:\n" + "\n".join([
                    f"- {item}" for item in memory_items
                ])
        
        # Create the prompt
        prompt = f"""
You are a memory extraction system. Your task is to analyze the following conversation and extract key information.
Focus on:
1. Important facts mentioned by the user
2. Preferences expressed by the user
3. Tasks or follow-ups mentioned
4. Contextual details that would be helpful to remember

{memory_context}

Here's the conversation:
{formatted_messages}

Provide your response in the following JSON format:
{{
  "key_points": [
    "Point 1",
    "Point 2",
    ...
  ],
  "user_preferences": [
    "Preference 1",
    "Preference 2",
    ...
  ],
  "tasks": [
    "Task 1",
    "Task 2",
    ...
  ],
  "context": "Brief summary of the conversation context"
}}
"""
        return prompt
    
    def _parse_memory_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured memory format"""
        try:
            # Try to find and extract JSON from the response
            response = response.strip()
            
            # Look for JSON opening/closing braces
            start_idx = response.find("{")
            end_idx = response.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx+1]
                memory_data = json.loads(json_str)
                
                # Ensure the expected structure
                expected_keys = ["key_points", "user_preferences", "tasks", "context"]
                for key in expected_keys:
                    if key not in memory_data:
                        memory_data[key] = [] if key != "context" else ""
                
                return memory_data
            else:
                # If no JSON found, create a basic structure with the response as context
                return {
                    "key_points": [],
                    "user_preferences": [],
                    "tasks": [],
                    "context": response[:200]  # Use first 200 chars as context
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse memory response as JSON: {str(e)}")
            # Fall back to a basic structure
            return {
                "key_points": [],
                "user_preferences": [],
                "tasks": [],
                "context": response[:200] if response else ""
            }
            
        except Exception as e:
            logger.error(f"Unexpected error parsing memory response: {str(e)}")
            return {
                "key_points": [],
                "user_preferences": [],
                "tasks": [],
                "context": ""
            }

# Create singleton instance
memory_extractor = MemoryExtractor()