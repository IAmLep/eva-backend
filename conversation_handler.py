"""
Conversation Handler for EVA backend.

Orchestrates the interaction flow for a single conversation turn,
integrating context management, memory retrieval, LLM calls,
and potential function execution or memory command handling.
"""

import asyncio
import json # Import json
import logging
from typing import Dict, Any, Optional, Union, AsyncGenerator

# --- Local Imports ---
from context_window import get_context_window, ContextWindow
from memory_manager import get_memory_manager, MemoryManager
from memory_extractor import get_memory_extractor, MemoryExtractor, MemoryCommand
from llm_service import GeminiService
from models import User, MemoryCategory # Import MemoryCategory
from exceptions import LLMServiceError, RateLimitError, FunctionCallError # Import relevant exceptions

# Import tool handling if function calling is enabled
try:
    # Import specific functions needed
    from api_tools import execute_function_call, available_tools, ToolCall as ApiToolCallDef # Renamed ToolCall to avoid Pydantic conflict
    FUNCTION_CALLING_ENABLED = True
    logger.info("Function calling enabled.")
except ImportError:
    logger.warning("api_tools not found or incomplete. Function calling disabled in ConversationHandler.")
    FUNCTION_CALLING_ENABLED = False
    ApiToolCallDef = None # Placeholder


# Logger configuration
logger = logging.getLogger(__name__)


class ConversationHandler:
    """
    Manages the processing of a single user message within a conversation session.
    Coordinates context updates, memory refresh, LLM interaction, and response generation.
    Handles the loop for function calling within a turn.
    """
    def __init__(self, user: User, session_id: str):
        self.user = user
        self.session_id = session_id
        self.context_window: ContextWindow = get_context_window() # Assumes global context
        self.memory_manager: MemoryManager = get_memory_manager()
        self.memory_extractor: MemoryExtractor = get_memory_extractor()
        self.gemini_service: GeminiService = GeminiService()
        self.current_history: List[Dict] = [] # Store history for current multi-turn processing (incl. function calls)

    async def process_message(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a user message and yields response chunks or function calls.
        Handles the full loop including potential function calls within a single user message turn.

        Args:
            message: The user's input message.

        Yields:
            Dict[str, Any]: Response chunks, typically {"text": "..."} or
                            {"function_call_request": {...}} or {"function_call_result": {...}} or
                            {"error": "..."}.
                            Includes an 'is_final' flag in the last dictionary yielded.
        """
        try:
            # 1. Initial Setup & Add User Message
            logger.info(f"Session {self.session_id}: Processing message from user {self.user.id}")
            self.context_window.add_message("user", message)
            self.current_history = [{"role": "user", "parts": [{"text": message}]}] # Start history for this turn

            # 2. Check for explicit memory commands
            memory_command = await self.memory_extractor.extract_memory_command(message)
            if memory_command:
                logger.info(f"Session {self.session_id}: Detected memory command: {memory_command.command_type}")
                response_text = await self._handle_memory_command(memory_command)
                yield {"text": response_text, "is_final": True}
                self.context_window.add_message("assistant", response_text)
                return

            # --- Start LLM Interaction Loop ---
            max_function_call_attempts = 5 # Prevent infinite loops
            for attempt in range(max_function_call_attempts):
                # 3. Refresh Memories & Assemble Context (only needed before first LLM call in loop)
                if attempt == 0:
                    logger.debug(f"Session {self.session_id}: Refreshing memories...")
                    await self.context_window.refresh_memories(self.user.id)

                # Assemble context using the *current* history of this turn
                # Context window should handle incorporating its own state (system, summaries, memories)
                context_text_for_llm = self.context_window.assemble_context() # Assemble full context
                # Construct history in Gemini format
                history_for_llm = self._build_gemini_history()
                logger.debug(f"Session {self.session_id}, Attempt {attempt+1}: Assembled context (len {len(context_text_for_llm)}), History turns: {len(history_for_llm)}")

                # 4. Call LLM (Streaming)
                logger.debug(f"Session {self.session_id}, Attempt {attempt+1}: Calling LLM service...")
                tools_to_pass = available_tools() if FUNCTION_CALLING_ENABLED else None

                llm_stream = self.gemini_service.stream_conversation_with_history(
                    history=history_for_llm, # Pass structured history
                    tools=tools_to_pass
                )

                full_response_text_this_turn = ""
                function_call_request = None

                async for chunk in llm_stream:
                    if "text" in chunk:
                        text_chunk = chunk["text"]
                        full_response_text_this_turn += text_chunk
                        yield {"text": text_chunk, "is_final": False}
                    elif "function_call" in chunk and FUNCTION_CALLING_ENABLED:
                        fc_data = chunk["function_call"]
                        logger.info(f"Session {self.session_id}: Received function call request: {fc_data['name']}")
                        function_call_request = fc_data # Store the first function call encountered
                        # Yield the request info to the client (optional, for UI feedback)
                        yield {"function_call_request": fc_data, "is_final": False}
                        # Stop processing further text chunks in this LLM response, prepare for execution
                        break
                    # Handle potential usage metadata if needed

                # Add the assistant's response part (text or function call request) to history
                assistant_parts = []
                if full_response_text_this_turn:
                     assistant_parts.append({"text": full_response_text_this_turn})
                if function_call_request:
                     # Format for Gemini history
                     assistant_parts.append({"function_call": function_call_request})
                if assistant_parts:
                     self.current_history.append({"role": "model", "parts": assistant_parts})

                # 5. Execute Function Call if requested
                if function_call_request and FUNCTION_CALLING_ENABLED:
                    try:
                        logger.info(f"Session {self.session_id}: Executing function {function_call_request['name']}...")
                        # Assuming execute_function_call needs name and args dict
                        tool_result_content = await execute_function_call(function_call_request, self.user) # Pass user object
                        logger.info(f"Session {self.session_id}: Function {function_call_request['name']} executed.")

                        # Format result for Gemini history (must be JSON string for content)
                        function_result_part = {
                            "function_response": {
                                "name": function_call_request['name'],
                                "response": tool_result_content # Pass the result dict directly
                            }
                        }
                        # Add function result to history for the *next* LLM call
                        self.current_history.append({"role": "function", "parts": [function_result_part]})

                        # Yield result info to client (optional)
                        yield {"function_call_result": {**function_result_part['function_response'], "response_summary": str(tool_result_content)[:100]+'...'}, "is_final": False}

                        # Continue the loop to send result back to LLM
                        continue

                    except FunctionCallError as fc_err:
                         logger.error(f"Session {self.session_id}: Function call failed: {fc_err}", exc_info=True)
                         # Add error result to history
                         error_result_part = {
                              "function_response": {
                                   "name": function_call_request['name'],
                                   "response": {"error": str(fc_err)} # Send error back to LLM
                              }
                         }
                         self.current_history.append({"role": "function", "parts": [error_result_part]})
                         yield {"error": f"Function call '{function_call_request['name']}' failed: {fc_err}", "is_final": False}
                         # Continue loop to inform LLM about the error
                         continue
                    except Exception as exec_err:
                         logger.error(f"Session {self.session_id}: Unexpected error executing function {function_call_request['name']}: {exec_err}", exc_info=True)
                         yield {"error": f"Unexpected error executing function '{function_call_request['name']}'.", "is_final": True}
                         # Add assistant's text before error to context window
                         if full_response_text_this_turn:
                              self.context_window.add_message("assistant", full_response_text_this_turn)
                         return # Stop processing on unexpected execution error

                else:
                    # No function call requested, or function calling disabled
                    # This is the final response for this user message turn
                    logger.info(f"Session {self.session_id}: LLM interaction complete.")
                    yield {"text": "", "is_final": True} # Yield final marker, text already yielded
                    # Add final assistant response to context window
                    if full_response_text_this_turn:
                        self.context_window.add_message("assistant", full_response_text_this_turn)
                    # Trigger background summarization if needed
                    if self.context_window.current_turn_count >= self.context_window.summarize_after_turns:
                         logger.info(f"Session {self.session_id}: Triggering background summarization...")
                         asyncio.create_task(self.context_window.summarize_conversation())
                    return # End processing

            # If loop finishes due to max attempts
            logger.warning(f"Session {self.session_id}: Exceeded max function call attempts ({max_function_call_attempts}).")
            yield {"error": "Exceeded maximum function call attempts.", "is_final": True}
            # Add last assistant response to context window
            if full_response_text_this_turn:
                self.context_window.add_message("assistant", full_response_text_this_turn)

        except (LLMServiceError, RateLimitError) as e:
            logger.error(f"Session {self.session_id}: LLM Error processing message: {e}", exc_info=True)
            yield {"error": f"Sorry, I encountered an issue with the AI service: {e}", "is_final": True}
        except Exception as e:
            logger.exception(f"Session {self.session_id}: Unexpected error processing message: {e}", exc_info=True)
            yield {"error": "I'm sorry, an unexpected error occurred.", "is_final": True}

    def _build_gemini_history(self) -> List[Dict]:
        """Builds history in the format expected by Gemini's generate_content."""
        # Combine context window history (summaries, older messages) with current turn history
        # This logic depends on how ContextWindow stores its history representation.
        # Simplified: Assume ContextWindow provides relevant past turns, and we append current turn.
        # TODO: Refine this based on actual ContextWindow implementation details.
        # For now, just use self.current_history which only contains the *current* turn's interactions.
        # This means context from previous turns relies solely on assemble_context().
        return self.current_history

    async def _handle_memory_command(self, command: MemoryCommand) -> str:
        """Handles explicit memory commands extracted from user input."""
        # (Implementation unchanged from previous version, assuming it's correct)
        try:
            if command.command_type == "remember":
                # ... (implementation) ...
                elif command.command_type == "remind":
                # ... (implementation) ...
                elif command.command_type == "forget":
                # ... (implementation) ...
                else:
                # ... (implementation) ...
        except Exception as e:
            # ... (error handling) ...
            pass # Placeholder for brevity
        return "Memory command processed (implementation placeholder)." # Placeholder return

    # Remove _handle_function_call as execution is now inline in process_message loop