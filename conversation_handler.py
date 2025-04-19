"""
Conversation Handler for EVA backend.

Orchestrates the interaction flow for a single conversation turn,
integrating context management, memory retrieval, LLM calls,
and potential function execution or memory command handling.
"""

import asyncio
import json
import logging # Ensure logging is imported
from typing import Dict, Any, Optional, Union, AsyncGenerator, List

# --- Logger Configuration (Moved Before Try/Except) ---
logger = logging.getLogger(__name__)

# --- Local Imports ---
from context_window import get_context_window, ContextWindow
from memory_manager import get_memory_manager, MemoryManager
from memory_extractor import get_memory_extractor, MemoryExtractor, MemoryCommand
from llm_service import GeminiService
from models import User, MemoryCategory
from exceptions import LLMServiceError, RateLimitError, FunctionCallError

# Import tool handling if function calling is enabled
try:
    # Attempt to import necessary components for function calling
    from api_tools import execute_function_call, available_tools, ToolCall as ApiToolCallDef
    FUNCTION_CALLING_ENABLED = True
    logger.info("Function calling enabled.") # Now logger is defined
except ImportError:
    # Log a warning if imports fail, indicating function calling is disabled
    logger.warning("api_tools not found or incomplete. Function calling disabled in ConversationHandler.") # Now logger is defined
    FUNCTION_CALLING_ENABLED = False
    ApiToolCallDef = None # Set placeholder to None if import fails

# --- Conversation Handler Class ---
class ConversationHandler:
    """
    Manages the processing of a single user message within a conversation session.
    Coordinates context updates, memory refresh, LLM interaction, and response generation.
    Handles the loop for function calling within a turn.
    """
    def __init__(self, user: User, session_id: str):
        """
        Initializes the ConversationHandler.

        Args:
            user: The authenticated user object.
            session_id: The unique identifier for the current conversation session.
        """
        self.user = user
        self.session_id = session_id
        # Get instances of dependencies (assumed to be singletons or managed elsewhere)
        self.context_window: ContextWindow = get_context_window()
        self.memory_manager: MemoryManager = get_memory_manager()
        self.memory_extractor: MemoryExtractor = get_memory_extractor()
        self.gemini_service: GeminiService = GeminiService()
        # Initialize history for the current processing turn (resets per message)
        self.current_history: List[Dict] = []
        logger.debug(f"ConversationHandler initialized for session {session_id}, user {user.id}")

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
            # 1. Initial Setup & Add User Message to Context and History
            logger.info(f"Session {self.session_id}: Processing message from user {self.user.id}")
            self.context_window.add_message("user", message)
            self.current_history = [{"role": "user", "parts": [{"text": message}]}] # Start Gemini history for this turn

            # 2. Check for explicit memory commands (e.g., "remember this:", "forget about X")
            memory_command = await self.memory_extractor.extract_memory_command(message)
            if memory_command:
                logger.info(f"Session {self.session_id}: Detected memory command: {memory_command.command_type}")
                response_text = await self._handle_memory_command(memory_command)
                # Yield the result and add to context window as assistant's final response
                yield {"text": response_text, "is_final": True}
                self.context_window.add_message("assistant", response_text)
                return # Stop processing as the command is handled

            # --- Start LLM Interaction Loop (Handles potential function calls) ---
            max_function_call_attempts = 5 # Limit attempts to prevent infinite loops
            for attempt in range(max_function_call_attempts):
                # 3. Refresh Memories & Assemble Context (only needed before first LLM call in loop)
                if attempt == 0:
                    # Retrieve relevant memories based on current context/message
                    logger.debug(f"Session {self.session_id}: Refreshing memories...")
                    await self.context_window.refresh_memories(self.user.id)

                # Assemble the full context string and history for the LLM
                context_text_for_llm = self.context_window.assemble_context() # For logging/debugging
                history_for_llm = self._build_gemini_history() # Get history in Gemini format
                logger.debug(f"Session {self.session_id}, Attempt {attempt+1}: Assembled context (len {len(context_text_for_llm)}), History turns: {len(history_for_llm)}")

                # 4. Call LLM Service (Streaming)
                logger.debug(f"Session {self.session_id}, Attempt {attempt+1}: Calling LLM service...")
                # Pass available tools only if function calling is enabled
                tools_to_pass = available_tools() if FUNCTION_CALLING_ENABLED else None

                llm_stream = self.gemini_service.stream_conversation_with_history(
                    history=history_for_llm,
                    tools=tools_to_pass
                )

                full_response_text_this_turn = ""
                function_call_request = None # Store the first function call request received

                # Process the stream from the LLM
                async for chunk in llm_stream:
                    if "text" in chunk:
                        text_chunk = chunk["text"]
                        full_response_text_this_turn += text_chunk
                        # Yield text chunks to the client
                        yield {"text": text_chunk, "is_final": False}
                    elif "function_call" in chunk and FUNCTION_CALLING_ENABLED:
                        # Received a function call request
                        fc_data = chunk["function_call"]
                        logger.info(f"Session {self.session_id}: Received function call request: {fc_data['name']}")
                        if function_call_request is None: # Only process the first one if multiple are sent (unlikely but possible)
                             function_call_request = fc_data
                        # Yield the request info to the client (optional, for UI feedback)
                        yield {"function_call_request": fc_data, "is_final": False}
                        # Stop processing further text chunks in this LLM response, prepare for execution
                        break # Exit chunk loop to execute the function call
                    # Handle potential usage metadata if needed (e.g., token counts)

                # Add the assistant's response part (text or function call request) to the current turn's history
                assistant_parts = []
                if full_response_text_this_turn:
                     assistant_parts.append({"text": full_response_text_this_turn})
                if function_call_request:
                     # Format for Gemini history
                     assistant_parts.append({"function_call": function_call_request})
                # Only add if there were parts (avoid empty model turns in history)
                if assistant_parts:
                     self.current_history.append({"role": "model", "parts": assistant_parts})

                # 5. Execute Function Call if one was requested and enabled
                if function_call_request and FUNCTION_CALLING_ENABLED:
                    try:
                        logger.info(f"Session {self.session_id}: Executing function {function_call_request['name']}...")
                        # Execute the function using the imported handler
                        tool_result_content = await execute_function_call(function_call_request, self.user) # Pass user object if needed by tools
                        logger.info(f"Session {self.session_id}: Function {function_call_request['name']} executed successfully.")

                        # Format the result for Gemini history (as a function_response part)
                        function_result_part = {
                            "function_response": {
                                "name": function_call_request['name'],
                                "response": tool_result_content # Pass the result dict/object directly
                            }
                        }
                        # Add the function result to history for the *next* LLM call
                        self.current_history.append({"role": "function", "parts": [function_result_part]})

                        # Yield result info to client (optional, for UI feedback)
                        yield {"function_call_result": {**function_result_part['function_response'], "response_summary": str(tool_result_content)[:100]+'...'}, "is_final": False}

                        # Continue the loop: send the function result back to the LLM
                        continue

                    except FunctionCallError as fc_err:
                         # Handle errors specifically raised during function execution
                         logger.error(f"Session {self.session_id}: Function call failed: {fc_err}", exc_info=True)
                         # Format the error result for Gemini history
                         error_result_part = {
                              "function_response": {
                                   "name": function_call_request['name'],
                                   "response": {"error": str(fc_err)} # Send structured error back to LLM
                              }
                         }
                         self.current_history.append({"role": "function", "parts": [error_result_part]})
                         # Yield error info to the client
                         yield {"error": f"Function call '{function_call_request['name']}' failed: {fc_err}", "is_final": False}
                         # Continue loop to inform LLM about the error
                         continue
                    except Exception as exec_err:
                         # Handle unexpected errors during function execution phase
                         logger.error(f"Session {self.session_id}: Unexpected error executing function {function_call_request['name']}: {exec_err}", exc_info=True)
                         yield {"error": f"Unexpected error executing function '{function_call_request['name']}'.", "is_final": True}
                         # Add assistant's text (if any) before error to context window
                         if full_response_text_this_turn:
                              self.context_window.add_message("assistant", full_response_text_this_turn)
                         return # Stop processing on unexpected execution error

                else:
                    # No function call was requested, or function calling is disabled
                    # This means the LLM's response (text) is the final response for this turn
                    logger.info(f"Session {self.session_id}: LLM interaction complete (no function call).")
                    # Yield a final marker, the full text was already yielded in chunks
                    yield {"text": "", "is_final": True}
                    # Add the complete assistant response text to the context window
                    if full_response_text_this_turn:
                        self.context_window.add_message("assistant", full_response_text_this_turn)
                    # Trigger background summarization if the conversation is long enough
                    # Check if summarization is enabled and threshold reached
                    if self.context_window.summarize_after_turns > 0 and \
                       self.context_window.current_turn_count >= self.context_window.summarize_after_turns:
                         logger.info(f"Session {self.session_id}: Triggering background summarization (turn {self.context_window.current_turn_count})...")
                         # Run summarization in the background without waiting
                         asyncio.create_task(self.context_window.summarize_conversation())
                    return # End processing for this user message

            # If the loop finishes because max_function_call_attempts was reached
            logger.warning(f"Session {self.session_id}: Exceeded max function call attempts ({max_function_call_attempts}).")
            yield {"error": "Exceeded maximum function call attempts.", "is_final": True}
            # Add the last assistant response text (if any) to the context window
            if full_response_text_this_turn:
                self.context_window.add_message("assistant", full_response_text_this_turn)

        except (LLMServiceError, RateLimitError) as e:
            # Handle known errors related to external services or limits
            logger.error(f"Session {self.session_id}: Service Error processing message: {e}", exc_info=True)
            yield {"error": f"Sorry, I encountered an issue: {e}", "is_final": True}
        except Exception as e:
            # Catch-all for any other unexpected errors during processing
            logger.exception(f"Session {self.session_id}: Unexpected error processing message: {e}", exc_info=True)
            yield {"error": "I'm sorry, an unexpected error occurred while processing your message.", "is_final": True}

    def _build_gemini_history(self) -> List[Dict]:
        """
        Builds the conversation history in the format expected by Gemini's generate_content.
        Currently uses only the history accumulated within the current `process_message` call.
        """
        # Placeholder: Use only the current turn's history for now.
        # TODO: Integrate with ContextWindow's stored history representation if needed.
        # This depends on how ContextWindow manages its internal state (e.g., if it returns
        # a list of past turns suitable for Gemini). For now, we rely on the context window's
        # `assemble_context()` method to provide historical context via memories/summaries,
        # and `self.current_history` handles the immediate back-and-forth of the current turn
        # (user message -> model response -> function call -> function response -> model response).
        return self.current_history

    async def _handle_memory_command(self, command: MemoryCommand) -> str:
        """
        Handles explicit memory commands (remember, remind, forget) extracted from user input.

        Args:
            command: The MemoryCommand object extracted by MemoryExtractor.

        Returns:
            A string response to be sent back to the user.
        """
        logger.debug(f"Handling memory command: {command.command_type} with content: '{command.content}'")
        response_message = "Sorry, I couldn't process that memory command." # Default error message

        try:
            if command.command_type == "remember":
                # Add the memory using the MemoryManager
                memory = await self.memory_manager.add_memory(
                    user_id=self.user.id,
                    session_id=self.session_id, # Link memory to session if needed
                    content=command.content,
                    category=command.category or MemoryCategory.GENERAL # Use extracted category or default
                )
                if memory:
                    response_message = f"Okay, I've remembered: \"{memory.content}\""
                    logger.info(f"Memory added for user {self.user.id}: {memory.id}")
                else:
                    response_message = "Sorry, I had trouble remembering that right now."
                    logger.error(f"Failed to add memory for user {self.user.id}. Content: {command.content}")

            elif command.command_type == "remind":
                # Search for memories related to the command content
                memories = await self.memory_manager.search_memories(
                    user_id=self.user.id,
                    query=command.content,
                    limit=5 # Limit the number of reminders shown
                )
                if memories:
                    # Format the found memories into a user-friendly list
                    reminders = "\n".join([f"- {m.content} (Added: {m.created_at.strftime('%Y-%m-%d')})" for m in memories])
                    response_message = f"Here's what I found related to \"{command.content}\":\n{reminders}"
                else:
                    response_message = f"I couldn't find any memories related to \"{command.content}\"."
                logger.info(f"Memory search performed for user {self.user.id}. Query: '{command.content}'. Found: {len(memories)}")

            elif command.command_type == "forget":
                 # Forgetting is complex. Simple approach: search and delete the most relevant match.
                 # A more robust approach might involve asking the user for confirmation if multiple matches exist.
                 memories_to_forget = await self.memory_manager.search_memories(
                     user_id=self.user.id,
                     query=command.content,
                     limit=1 # Find the single most relevant memory matching the query
                 )
                 if memories_to_forget:
                     memory_to_forget = memories_to_forget[0]
                     deleted = await self.memory_manager.delete_memory(self.user.id, memory_to_forget.id)
                     if deleted:
                         response_message = f"Okay, I've forgotten the memory: \"{memory_to_forget.content}\"."
                         logger.info(f"Memory deleted for user {self.user.id}: {memory_to_forget.id}")
                     else:
                         response_message = f"Sorry, I tried but couldn't forget the memory: \"{memory_to_forget.content}\"."
                         logger.error(f"Failed to delete memory {memory_to_forget.id} for user {self.user.id}")
                 else:
                     response_message = f"I couldn't find a specific memory matching \"{command.content}\" to forget."
                     logger.warning(f"Memory forget command for user {self.user.id} found no match for: '{command.content}'")

            else:
                # Handle any unexpected command types if MemoryExtractor could produce others
                logger.warning(f"Received unknown memory command type: {command.command_type}")
                response_message = "I'm not sure how to handle that memory command."

        except Exception as e:
            # Catch-all for errors during memory operations
            logger.exception(f"Error handling memory command '{command.command_type}' for user {self.user.id}: {e}", exc_info=True)
            response_message = f"Sorry, an error occurred while processing the memory command: {e}"

        return response_message