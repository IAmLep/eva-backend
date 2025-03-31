import logging
from typing import List, Dict, Any, Optional
from firestore_manager import FirestoreManager

logger = logging.getLogger(__name__)

class MemoryExtractor:
    """Extract and manage conversation memory"""
    
    def __init__(self):
        self.firestore = FirestoreManager()
    
    async def extract_key_info(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Extract key information from a conversation's history
        and store it as a memory summary
        """
        # Get the conversation
        conversation = await self.firestore.get_conversation(conversation_id, user_id)
        if not conversation:
            logger.warning(f"Conversation {conversation_id} not found for extraction")
            return None
        
        # Get the conversation history
        messages = conversation.get("messages", [])
        if not messages or len(messages) < 2:  # Need at least a user query and a response
            logger.info(f"Not enough messages in conversation {conversation_id} for extraction")
            return None
        
        # Extract the last 10 messages for context
        recent_messages = messages[-10:] if len(messages) > 10 else messages
        
        # Format the conversation for the LLM to extract key information
        formatted_conversation = "\n".join([
            f"{'User' if msg.get('role') == 'user' else 'Assistant'}: {msg.get('content', '')}"
            for msg in recent_messages
        ])
        
        # Prepare prompt for extracting key information
        prompt = f"""
        Based on the following conversation, extract and summarize key information 
        that should be remembered for future interactions. Focus on facts, preferences, 
        and important details about the user.
        
        CONVERSATION:
        {formatted_conversation}
        
        EXTRACTION:
        """
        
        # Get LLM service to extract information
        from llm_service import LLMService
        llm = LLMService()
        
        try:
            # Call LLM to extract information
            extraction = await llm.generate_text(prompt, max_tokens=200)
            
            # Store the extraction as a memory summary
            memory_data = {
                "conversation_id": conversation_id,
                "summary": extraction,
                "created_at": self.firestore.server_timestamp(),
                "extracted_from_messages": len(messages)
            }
            
            # Store in Firestore
            memory_ref = self.firestore.db.collection("conversation_summaries").document(conversation_id)
            memory_ref.set(memory_data, merge=True)
            
            return memory_data
            
        except Exception as e:
            logger.error(f"Error extracting memory from conversation {conversation_id}: {str(e)}")
            return None
    
    async def get_conversation_memory(self, conversation_id: str, user_id: str) -> str:
        """Get the memory summary for a conversation"""
        # Check if we have a memory summary stored
        memory_ref = self.firestore.db.collection("conversation_summaries").document(conversation_id)
        memory = memory_ref.get()
        
        if memory.exists:
            # Verify ownership through conversation
            conversation = await self.firestore.get_conversation(conversation_id, user_id)
            if not conversation or conversation.get("user_id") != user_id:
                logger.warning(f"User {user_id} attempted to access memory for conversation {conversation_id}")
                return ""
            
            memory_data = memory.to_dict()
            return memory_data.get("summary", "")
        
        # If no summary exists, try to create one
        extracted = await self.extract_key_info(conversation_id, user_id)
        if extracted:
            return extracted.get("summary", "")
        
        return ""