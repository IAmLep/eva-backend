"""
Conversation Handler for EVA backend.

Orchestrates the interaction flow for a single conversation turn,
integrating context management, memory retrieval, LLM calls,
and potential function execution or memory command handling.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Union, AsyncGenerator

# --- Local Imports ---
from context_window import get_context_window, ContextWindow
from memory_manager import get_memory_manager, MemoryManager
from memory_extractor import get_memory_extractor, MemoryExtractor, MemoryCommand # Import MemoryCommand
from llm_service import GeminiService
from models import User
from exceptions import LLMServiceError, RateLimitError # Import relevant exceptions
# Import tool handling if function calling is enabled
try:
    from api_tools import execute_function_call, ToolCall, available_tools, ToolFunction
    FUNCTION_CALLING_ENABLED = True
except ImportError:
    logger.warning("api_tools not found or incomplete. Function calling disabled in ConversationHandler.")
    FUNCTION_CALLING_ENABLED = False
    # Define placeholders if needed, though might not be used if disabled
    class ToolCall: pass
    class ToolFunction: pass


# Logger configuration
logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    Manages the processing of a single user message within a conversation session.

    Coordinates context updates, memory refresh, LLM interaction, and response generation.
    Designed to be used per-request or per-WebSocket message, potentially holding
    short-term state for a multi-turn interaction if needed (though long-term state
    is in ContextWindow and DB).
    """
    def __init__(self, user: User, session_id: str):
        """
        Initialize the conversation handler for a specific user and session.

        Args:
            user: The user participating in the conversation.
            session_id: A unique identifier for the current conversation session.
        """
        self.user = user
        self.session_id = session_id # Store session ID for logging/context
        # Get singleton instances of managers
        self.context_window: ContextWindow = get_context_window() # Assuming global context for now
        self.memory_manager: MemoryManager = get_memory_manager()
        self.memory_extractor: MemoryExtractor = get_memory_extractor()
        self.gemini_service: GeminiService = GeminiService()
        # TODO: For true multi-session support, ContextWindow might need to be session-specific.
        # This current setup implies a single, shared context window which might not be desired.
        # If ContextWindow is session-specific, it should be created here:
        # self.context_window = ContextWindow() # Create new context per handler instance

    async def process_message(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a user message and yields response chunks or function calls.

        Args:
            message: The user's input message.

        Yields:
            Dict[str, Any]: Response chunks, typically {"text": "..."} or
                            {"function_call": {"name": "...", "args": {...}}} or
                            {"error": "..."}.
                            Includes an 'is_final' flag in the last dictionary yielded.
        """
        full_response_text = ""
        final_chunk = {}
        try:
            # 1. Add user message to context
            logger.info(f"Session {self.session_id}: Processing message from user {self.user.id}")
            self.context_window.add_message("user", message)

            # 2. Check for explicit memory commands (using MemoryExtractor)
            logger.debug(f"Session {self.session_id}: Checking for memory commands in: '{message[:50]}...'")
            memory_command = await self.memory_extractor.extract_memory_command(message)
            if memory_command:
                logger.info(f"Session {self.session_id}: Detected memory command: {memory_command.command_type}")
                response_text = await self._handle_memory_command(memory_command)
                final_chunk = {"text": response_text, "is_final": True}
                yield final_chunk
                # Add confirmation message to context *after* yielding response
                self.context_window.add_message("assistant", response_text)
                return # End processing for this message

            # 3. Refresh relevant memories (based on updated context)
            logger.debug(f"Session {self.session_id}: Refreshing memories...")
            await self.context_window.refresh_memories(self.user.id)

            # 4. Assemble context for LLM
            context_text = self.context_window.assemble_context()
            logger.debug(f"Session {self.session_id}: Assembled context (length: {len(context_text)} chars)")
            # logger.debug(f"Context Snippet:\n---\n{context_text[:500]}...\n---") # Log snippet if needed

            # 5. Call LLM (Streaming) & Handle Response/Function Calls
            logger.debug(f"Session {self.session_id}: Calling LLM service...")
            tools_to_pass = available_tools() if FUNCTION_CALLING_ENABLED else None

            llm_stream = self.gemini_service.stream_conversation(
                context=context_text,
                tools=tools_to_pass
            )

            async for chunk in llm_stream:
                if "text" in chunk:
                    text_chunk = chunk["text"]
                    full_response_text += text_chunk
                    yield {"text": text_chunk, "is_final": False}
                elif "function_call" in chunk and FUNCTION_CALLING_ENABLED:
                    fc_data = chunk["function_call"]
                    logger.info(f"Session {self.session_id}: Received function call request: {fc_data['name']}")
                    # Yield the function call request to the caller (e.g., websocket manager)
                    # The caller should execute it and potentially send result back for another LLM turn
                    # For now, we just yield it. A more complex flow would handle the execution here.
                    yield {"function_call": fc_data, "is_final": False}
                    # TODO: Implement logic to wait for function result and continue generation
                    # This often requires breaking the streaming loop or handling multiple turns.
                    # Simplified: Assume function call ends the current response turn for now.
                    final_chunk = {"text": full_response_text, "is_final": True, "function_call_pending": fc_data} # Mark final but indicate pending call
                    yield final_chunk
                    # Add assistant's partial response (before function call) to context
                    if full_response_text:
                         self.context_window.add_message("assistant", full_response_text)
                    # Add placeholder for function call in context? Maybe not needed until result.
                    return # End processing after function call for now

                # Handle potential usage metadata if yielded separately
                # elif "usage_metadata" in chunk:
                #     logger.debug(f"Session {self.session_id}: Received usage metadata: {chunk['usage_metadata']}")

            # 6. If streaming finished without function call, yield final text chunk
            if not final_chunk: # Ensure we haven't already yielded a final chunk
                final_chunk = {"text": full_response_text, "is_final": True}
                yield final_chunk

            # 7. Add complete assistant response to context
            if full_response_text:
                logger.debug(f"Session {self.session_id}: Adding assistant response to context.")
                self.context_window.add_message("assistant", full_response_text)

            # 8. Trigger background summarization if needed
            # Check *after* adding the assistant's response
            if self.context_window.current_turn_count >= self.context_window.summarize_after_turns:
                 logger.info(f"Session {self.session_id}: Triggering background summarization...")
                 # Run summarization in the background without blocking the response flow
                 asyncio.create_task(self.context_window.summarize_conversation())


        except (LLMServiceError, RateLimitError) as e:
            logger.error(f"Session {self.session_id}: LLM Error processing message: {e}", exc_info=True)
            yield {"error": f"Sorry, I encountered an issue with the AI service: {e}", "is_final": True}
        except Exception as e:
            logger.exception(f"Session {self.session_id}: Unexpected error processing message: {e}", exc_info=True)
            yield {"error": "I'm sorry, an unexpected error occurred.", "is_final": True}


    async def _handle_memory_command(self, command: MemoryCommand) -> str:
        """Handles explicit memory commands extracted from user input."""
        try:
            if command.command_type == "remember":
                if not command.content: return "What should I remember?"
                # Use default category if not provided by extractor
                category = command.category or MemoryCategory.FACT
                await self.memory_manager.create_core_memory(
                    user_id=self.user.id,
                    content=command.content,
                    category=category,
                    entity=command.entity,
                    importance=7 # Default importance for explicit remember
                )
                logger.info(f"Session {self.session_id}: Handled 'remember' command.")
                return f"Okay, I'll remember that: '{command.content[:50]}...'"

            elif command.command_type == "remind":
                if not command.content: return "What should I remind you about?"
                if command.event_time:
                    await self.memory_manager.create_event_memory(
                        user_id=self.user.id,
                        content=command.content,
                        event_time=command.event_time,
                        expiration=command.expiration # Pass expiration if extractor provided it
                    )
                    event_time_str = command.event_time.strftime("%Y-%m-%d %H:%M") # Format for response
                    logger.info(f"Session {self.session_id}: Handled 'remind' command for {event_time_str}.")
                    return f"Okay, I'll remind you at {event_time_str}: '{command.content[:50]}...'"
                else:
                    logger.warning(f"Session {self.session_id}: 'remind' command missing event time.")
                    # Ask LLM to clarify? Or just respond directly.
                    return "When should I remind you about that?"

            elif command.command_type == "forget":
                 if not command.content: return "What should I forget?"
                 # Forgetting is complex: need to find the memory first.
                 # Simple approach: Search memories by content/entity.
                 logger.info(f"Session {self.session_id}: Handling 'forget' command for content: '{command.content[:50]}...'")
                 # Use get_relevant_memories to find potential matches
                 matches, _ = await asyncio.gather(
                      self.memory_manager.get_relevant_memories(
                           user_id=self.user.id, query=command.content, limit=1
                      )
                      # Add other searches if needed (e.g., exact content match)
                 )
                 # relevant_memories = await self.memory_manager.get_relevant_memories(
                 #      user_id=self.user.id, query=command.content, limit=1
                 # )
                 if matches:
                      memory_to_forget, relevance = matches[0]
                      # Add a confirmation step in a real scenario
                      deleted = await self.memory_manager.delete_memory(memory_to_forget.memory_id, self.user.id)
                      if deleted:
                           return f"Okay, I've forgotten the memory about: '{memory_to_forget.content[:50]}...'"
                      else:
                           return "I found a related memory, but couldn't forget it."
                 else:
                      return "I couldn't find a specific memory matching that description to forget."

            else:
                logger.warning(f"Session {self.session_id}: Received unknown memory command type: {command.command_type}")
                return "I'm not sure how to handle that memory request."

        except Exception as e:
            logger.error(f"Session {self.session_id}: Error handling memory command '{command.command_type}': {e}", exc_info=True)
            return f"Sorry, I encountered an error trying to manage my memory: {e}"

    # --- Function Calling Handling (Example, needs integration with execution flow) ---
    async def _handle_function_call(self, tool_call_data: Dict) -> Dict:
         """Placeholder for executing a function call and getting result."""
         if not FUNCTION_CALLING_ENABLED:
              logger.error("Function call requested but feature is disabled.")
              return {"error": "Function calling is not enabled."}

         try:
              # Assume tool_call_data has {'name': str, 'args': dict}
              # Map this back to ToolCall if necessary or use directly
              # This requires `execute_function_call` to exist and work with this dict format
              logger.info(f"Session {self.session_id}: Executing function {tool_call_data['name']}...")
              # NOTE: execute_function_call needs access to the current User object
              result = await execute_function_call(tool_call_data, self.user) # Pass user
              logger.info(f"Session {self.session_id}: Function {tool_call_data['name']} executed.")
              # Format result for sending back to LLM
              return {
                   "tool_name": tool_call_data['name'],
                   "content": json.dumps(result) # Gemini expects function result content as JSON string
              }
         except Exception as e:
              logger.error(f"Session {self.session_id}: Error executing function {tool_call_data['name']}: {e}", exc_info=True)
              return {
                   "tool_name": tool_call_data['name'],
                   "content": json.dumps({"error": f"Execution failed: {e}"})
              }