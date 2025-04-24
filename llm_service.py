"""
LLM Service module for EVA backend.

Integrates with the configured LLM provider (Gemini) for text generation,
streaming responses, and potentially function calling. Handles API key
configuration, error handling, and basic mocking.
"""

import asyncio
import base64
import json
import logging
import os
import uuid
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncGenerator

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError, ResourceExhausted, PermissionDenied, Unauthenticated
from pydantic import BaseModel, Field

# --- Local Imports ---
from cache_manager import cached
from config import settings
from exceptions import LLMServiceError, RateLimitError, AuthenticationError, ConfigurationError
from models import Memory # Keep Memory import if used elsewhere, maybe not needed directly here

# Corrected: Logger configuration should be defined BEFORE it's used below
logger = logging.getLogger(__name__)

# Assuming ToolFunction and ToolCall definitions are primarily used by api_tools.py
# but might be needed here if directly handling function call results.
# Define placeholders if api_tools.py or similar isn't providing them yet.
try:
    from api_tools import ToolFunction, ToolCall # Try importing if api_tools defines them
except ImportError:
    # Corrected: Now the logger exists when this warning is called
    logger.warning("ToolFunction/ToolCall not found in api_tools, using placeholder definitions.")
    class ToolFunction(BaseModel):
        name: str
        description: str
        # Use a more flexible definition if Parameter schema isn't fixed
        parameters: List[Dict[str, Any]] # Simplified parameters

    class ToolCall(BaseModel):
        # Use more flexible definitions if structure isn't fixed
        function: Dict[str, Any] # e.g., {"name": "...", "description": "..."}
        args: Dict[str, Any]


# --- Gemini API Type Imports ---
# Consolidate type imports with fallbacks
try:
    from google.generativeai.types import FunctionDeclaration, Tool, GenerationConfig, Content, Part
    logger.debug("Successfully imported Gemini types from google.generativeai.types")
except ImportError:
    try:
        # Fallback for slightly older versions if structure changed
        from google.generativeai.types.generation_types import FunctionDeclaration, Tool, GenerationConfig
        from google.generativeai.types.content_types import Content, Part
        logger.debug("Successfully imported Gemini types from generation_types/content_types")
    except ImportError:
        logger.warning("Could not import specific Gemini types (FunctionDeclaration, Tool, GenerationConfig, Content, Part). Using dict-based fallbacks.")
        # Define placeholders if imports fail
        FunctionDeclaration = dict
        Tool = dict
        GenerationConfig = dict
        Content = dict
        Part = dict


class StreamProgress(BaseModel):
    """Model for tracking streaming response progress (optional)."""
    content: str = ""
    is_complete: bool = False
    tool_calls: List[Dict] = Field(default_factory=list) # Use dict if ToolCall isn't reliably defined
    token_count: int = 0


class GeminiService:
    """
    Service for interacting with the Google Gemini API.

    Handles API key configuration, text generation, streaming,
    basic function calling setup, and error management.
    """

    def __init__(self):
        """Initialize Gemini service using settings."""
        self.settings = settings
        self.api_key = self.settings.GEMINI_API_KEY
        self.model_name = self.settings.GEMINI_MODEL

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found in settings. Using mock responses.")
            self.use_mock = True
            self.client = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                # Test configuration by creating a model instance
                self.client = genai.GenerativeModel(self.model_name)
                self.use_mock = False
                logger.info(f"Gemini service initialized successfully for model: {self.model_name}")
            except Exception as e:
                logger.exception(f"Failed to configure Gemini API: {e}. Falling back to mock responses.", exc_info=e)
                self.use_mock = True
                self.client = None

    def _get_generation_config(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> Union[GenerationConfig, Dict]:
        """Creates a GenerationConfig object or a fallback dictionary."""
        # Use instance defaults or provided values
        temp_to_use = temperature if temperature is not None else self.settings.LLM_TEMPERATURE
        tokens_to_use = max_tokens if max_tokens is not None else self.settings.LLM_MAX_TOKENS

        # Ensure max_tokens doesn't exceed a reasonable limit if needed
        # tokens_to_use = min(tokens_to_use, 8192) # Example safety limit

        try:
            # Use the imported GenerationConfig class if available
            if GenerationConfig is not dict:
                return GenerationConfig(
                    temperature=temp_to_use,
                    max_output_tokens=tokens_to_use,
                    # Add other params as needed, checking compatibility with the specific model
                    # top_p=0.95,
                    # top_k=40,
                )
            else:
                # Fallback to dictionary if class import failed
                logger.debug("GenerationConfig type not available, using dict fallback.")
                return {
                    "temperature": temp_to_use,
                    "max_output_tokens": tokens_to_use,
                }
        except Exception as e:
             logger.warning(f"Failed to create GenerationConfig object, using fallback dict: {e}")
             # Fallback to dictionary
             return {
                "temperature": temp_to_use,
                "max_output_tokens": tokens_to_use,
             }

    def _prepare_tools(self, tools: Optional[List[ToolFunction]] = None) -> Optional[List[Union[Tool, dict]]]:
        """Converts internal ToolFunction list to Gemini API Tool format or fallback dict."""
        if not tools:
            return None

        # Check if proper types were imported
        use_fallback_types = FunctionDeclaration is dict or Tool is dict

        gemini_tools = []
        try:
            for tool_func in tools:
                # Convert Pydantic parameters to OpenAPI JSON Schema format
                properties = {}
                required = []
                # Handle parameters which might be Dict or a Pydantic model if ToolFunction wasn't imported
                params_to_iterate = []
                if isinstance(tool_func, ToolFunction) and hasattr(tool_func, 'parameters'):
                    params_to_iterate = tool_func.parameters
                elif isinstance(tool_func, dict) and 'parameters' in tool_func:
                     # Basic handling if ToolFunction is just a dict
                     params_to_iterate = [p for p in tool_func['parameters'] if isinstance(p, dict)]


                for param in params_to_iterate:
                    # Need to access dict keys if param is not a Pydantic model
                    param_name = param.get('name') if isinstance(param, dict) else param.name
                    param_type = param.get('type') if isinstance(param, dict) else param.type
                    param_desc = param.get('description', '') if isinstance(param, dict) else param.description
                    param_enum = param.get('enum') if isinstance(param, dict) else getattr(param, 'enum', None)
                    param_req = param.get('required', False) if isinstance(param, dict) else param.required

                    if not param_name or not param_type: continue # Skip invalid params

                    param_schema = {
                        "type": param_type,
                        "description": param_desc
                    }
                    if param_enum:
                        param_schema["enum"] = param_enum
                    properties[param_name] = param_schema
                    if param_req:
                        required.append(param_name)

                openapi_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }

                tool_name = tool_func.name if isinstance(tool_func, ToolFunction) else tool_func.get('name', 'unknown_tool')
                tool_desc = tool_func.description if isinstance(tool_func, ToolFunction) else tool_func.get('description', '')

                if use_fallback_types:
                     logger.debug(f"Using dict fallback for FunctionDeclaration/Tool for tool: {tool_name}")
                     function_decl_data = {
                         "name": tool_name,
                         "description": tool_desc,
                         "parameters": openapi_schema
                     }
                     gemini_tools.append({"function_declarations": [function_decl_data]})
                else:
                    function_decl = FunctionDeclaration(
                        name=tool_name,
                        description=tool_desc,
                        parameters=openapi_schema
                    )
                    gemini_tools.append(Tool(function_declarations=[function_decl])) # Gemini expects a list

            return gemini_tools if gemini_tools else None
        except Exception as e:
            logger.error(f"Failed to prepare tools for Gemini API: {e}", exc_info=True)
            return None # Return None if preparation fails

    async def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Union[ToolFunction, Dict]]] = None # Accept dict if ToolFunction fails import
    ) -> Tuple[str, Dict[str, int], Optional[List[Dict]]]:
        """
        Generate text using the configured Gemini model.
        (Docstring remains the same)
        """
        if self.use_mock:
            # Pass tools to mock, ensuring it handles potential dict format
            mock_tools_list = []
            if tools:
                for t in tools:
                    if isinstance(t, dict): mock_tools_list.append(t)
                    elif hasattr(t, 'dict'): mock_tools_list.append(t.dict()) # Convert Pydantic model
            return self._mock_generate_text(prompt, mock_tools_list)

        if not self.client:
             raise ConfigurationError("Gemini client is not initialized. Check API key and configuration.")

        generation_config = self._get_generation_config(temperature, max_tokens)
        gemini_tools = self._prepare_tools(tools) # Prepare tools handles potential dict format

        logger.debug(f"Generating text with model {self.model_name}. Prompt length: {len(prompt)}")
        try:
            # Ensure generation_config is passed correctly whether it's an object or dict
            config_arg = generation_config if not isinstance(generation_config, dict) else None
            config_kwarg = generation_config if isinstance(generation_config, dict) else {}

            response = await asyncio.to_thread(
                self.client.generate_content,
                contents=[prompt], # Pass prompt as contents list
                generation_config=config_arg,
                tools=gemini_tools if gemini_tools else None,
                **config_kwarg # Pass as kwargs if it's a dict
            )

            # --- Process Response ---
            generated_text = ""
            tool_calls_list = []

            # Check for response content and parts
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        generated_text += part.text
                    elif hasattr(part, 'function_call'):
                        # Handle function call extraction
                        fc = part.function_call
                        # Ensure args are a dictionary
                        try:
                            # fc.args might be a RepeatedScalarCompositeContainer or similar, cast explicitly
                            args_dict = dict(fc.args)
                        except Exception as args_ex:
                            logger.error(f"Could not convert function call args to dict for {fc.name}. Args type: {type(fc.args)}, Error: {args_ex}. Args: {fc.args}")
                            args_dict = {}

                        tool_calls_list.append({
                            "name": fc.name,
                            "args": args_dict,
                        })

            # --- Token Usage ---
            usage_metadata = getattr(response, 'usage_metadata', None)
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0) if usage_metadata else 0
            candidates_tokens = getattr(usage_metadata, 'candidates_token_count', 0) if usage_metadata else 0
            total_tokens = getattr(usage_metadata, 'total_token_count', 0) if usage_metadata else (prompt_tokens + candidates_tokens)

            token_info = {
                "input_tokens": prompt_tokens,
                "output_tokens": candidates_tokens,
                "total_tokens": total_tokens
            }
            logger.debug(f"Gemini response received. Tokens: {token_info}. Tool calls: {len(tool_calls_list)}")

            return generated_text, token_info, tool_calls_list if tool_calls_list else None

        # --- Error Handling (remains the same) ---
        except (ResourceExhausted, GoogleAPIError) as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning(f"Gemini API rate limit exceeded: {e}")
                raise RateLimitError(f"Gemini API rate limit exceeded: {e}")
            else:
                logger.error(f"Gemini API error during text generation: {e}", exc_info=True)
                raise LLMServiceError(f"Gemini API error: {e}")
        except (PermissionDenied, Unauthenticated) as e:
            logger.error(f"Gemini API authentication/permission error: {e}", exc_info=True)
            raise AuthenticationError(f"Gemini API authentication error: {e}")
        except Exception as e:
            logger.exception(f"Unexpected error during Gemini text generation: {e}", exc_info=True)
            raise LLMServiceError(f"Unexpected error generating text: {e}")

    async def stream_conversation(
        self,
        context: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Union[ToolFunction, Dict]]] = None, # Accept dicts too
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream conversation response chunks from Gemini.
        (Docstring remains the same)
        """
        if self.use_mock:
            # Pass tools to mock, ensuring it handles potential dict format
            mock_tools_list = []
            if tools:
                for t in tools:
                    if isinstance(t, dict): mock_tools_list.append(t)
                    elif hasattr(t, 'dict'): mock_tools_list.append(t.dict())
            async for chunk in self._mock_stream_conversation(context, mock_tools_list):
                yield chunk
            return

        if not self.client:
            raise ConfigurationError("Gemini client is not initialized. Check API key and configuration.")

        generation_config = self._get_generation_config(temperature, max_tokens)
        gemini_tools = self._prepare_tools(tools) # Handles potential dict format

        logger.debug(f"Streaming conversation with model {self.model_name}. Context length: {len(context)}")
        try:
            # Ensure generation_config is passed correctly whether it's an object or dict
            config_arg = generation_config if not isinstance(generation_config, dict) else None
            config_kwarg = generation_config if isinstance(generation_config, dict) else {}

            # Start the streaming generation
            stream = await asyncio.to_thread(
                self.client.generate_content,
                contents=[context],
                generation_config=config_arg,
                tools=gemini_tools if gemini_tools else None,
                stream=True,
                 **config_kwarg # Pass as kwargs if it's a dict
            )

            # Process the stream
            async for chunk in stream:
                # Check for potential errors in the stream chunk itself (less common but possible)
                if hasattr(chunk, 'prompt_feedback') and getattr(chunk.prompt_feedback, 'block_reason', None):
                     block_reason = chunk.prompt_feedback.block_reason
                     logger.error(f"Gemini stream blocked. Reason: {block_reason}")
                     raise LLMServiceError(f"Content blocked by API: {block_reason}")

                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            yield {"text": part.text}
                        elif hasattr(part, 'function_call'):
                            fc = part.function_call
                            try:
                                # Explicitly convert args to dict
                                args_dict = dict(fc.args)
                            except Exception as args_ex:
                                logger.error(f"Could not convert streaming function call args to dict for {fc.name}. Args type: {type(fc.args)}, Error: {args_ex}. Args: {fc.args}")
                                args_dict = {}
                            yield {"function_call": {"name": fc.name, "args": args_dict}}
                # Include usage metadata if available (often in the last chunk)
                usage_metadata = getattr(chunk, 'usage_metadata', None)
                if usage_metadata:
                     token_info = {
                         "input_tokens": usage_metadata.prompt_token_count,
                         "output_tokens": usage_metadata.candidates_token_count,
                         "total_tokens": usage_metadata.total_token_count
                     }
                     yield {"usage_metadata": token_info} # Yield token info

        # --- Error Handling (remains the same) ---
        except (ResourceExhausted, GoogleAPIError) as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                logger.warning(f"Gemini API rate limit exceeded during streaming: {e}")
                raise RateLimitError(f"Gemini API rate limit exceeded: {e}")
            else:
                logger.error(f"Gemini API error during streaming: {e}", exc_info=True)
                raise LLMServiceError(f"Gemini API streaming error: {e}")
        except (PermissionDenied, Unauthenticated) as e:
            logger.error(f"Gemini API authentication/permission error during streaming: {e}", exc_info=True)
            raise AuthenticationError(f"Gemini API authentication error: {e}")
        except LLMServiceError: # Re-raise errors caught within the stream loop
             raise
        except Exception as e:
            logger.exception(f"Unexpected error during Gemini streaming: {e}", exc_info=True)
            raise LLMServiceError(f"Unexpected error streaming conversation: {e}")

    # --- Mock Methods ---
    def _mock_generate_text(
        self, prompt: str, tools: Optional[List[Dict]] = None # Expect dicts now
    ) -> Tuple[str, Dict[str, int], Optional[List[Dict]]]:
        logger.warning("Using mock Gemini text generation")
        response_text = f"Mock response for prompt: '{prompt[:50]}...'"
        token_info = {"input_tokens": len(prompt)//4, "output_tokens": 20, "total_tokens": len(prompt)//4 + 20}
        tool_calls = None

        # Simple mock function call example
        if tools and "weather" in prompt.lower():
             # Find tool by name in the list of dicts
             weather_tool = next((t for t in tools if t.get('name') == "get_weather"), None)
             if weather_tool:
                  response_text = "Okay, I can get the weather for you."
                  tool_calls = [{"name": "get_weather", "args": {"location": "San Francisco"}}]

        return response_text, token_info, tool_calls

    async def _mock_stream_conversation(self, context: str, tools: Optional[List[Dict]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        logger.warning("Using mock Gemini conversation streaming")
        response = f"Mock stream response for context: '{context[:30]}...'. This is chunked."
        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.05) # Simulate delay
            yield {"text": word + (" " if i < len(words) - 1 else "")}

        # Optionally yield a mock function call at the end if tools were provided
        if tools:
             mock_tool = next((t for t in tools if 'mock' in t.get('name', '')), None)
             if mock_tool:
                  await asyncio.sleep(0.1)
                  yield {"function_call": {"name": mock_tool['name'], "args": {"param1": "mock_value"}}}

        # Yield mock usage data at the very end
        await asyncio.sleep(0.01)
        yield {"usage_metadata": {"input_tokens": len(context)//4, "output_tokens": len(words), "total_tokens": len(context)//4 + len(words)}}


# --- Helper Functions (Placeholders - Transcribe/Generate Voice) ---
# These remain unchanged as they are placeholders outside the core GeminiService logic

async def transcribe_audio(audio_data: bytes) -> str:
    """
    Transcribe audio data to text using an appropriate service (Placeholder).
    NOTE: Gemini API itself might have audio input capabilities, investigate that.
          This is a placeholder and doesn't call Gemini.
    """
    logger.info(f"Received audio data for transcription: {len(audio_data)} bytes")
    if not audio_data: return ""
    await asyncio.sleep(0.2) # Simulate processing time
    mock_transcription = f"Mock transcription of {len(audio_data)} bytes of audio."
    if len(audio_data) > 50000: mock_transcription += " It seems like a longer recording."
    logger.info(f"Mock transcription result: '{mock_transcription}'")
    return mock_transcription

async def generate_voice_response(text: str) -> bytes:
    """
    Generate voice (audio bytes) from text using an appropriate service (Placeholder).
    NOTE: This is a placeholder and doesn't call Gemini or a TTS service.
    """
    logger.info(f"Generating mock voice response for text: '{text[:50]}...'")
    await asyncio.sleep(0.1) # Simulate processing time
    mock_audio = f"AUDIO_FOR:[{text}]".encode('utf-8') * 3
    return mock_audio