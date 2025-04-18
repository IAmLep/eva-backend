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
from config import get_settings
from exceptions import LLMServiceError, RateLimitError, AuthenticationError, ConfigurationError
from models import Memory # Keep Memory import if used elsewhere, maybe not needed directly here
# Assuming ToolFunction and ToolCall definitions are primarily used by api_tools.py
# but might be needed here if directly handling function call results.
# Define placeholders if api_tools.py or similar isn't providing them yet.
try:
    from api_tools import ToolFunction, ToolCall # Try importing if api_tools defines them
except ImportError:
    logger.warning("ToolFunction/ToolCall not found in api_tools, using placeholder definitions.")
    class ToolFunction(BaseModel):
        name: str
        description: str
        parameters: List[Dict[str, Any]] # Simplified parameters

    class ToolCall(BaseModel):
        function: ToolFunction
        args: Dict[str, Any]


# Logger configuration
logger = logging.getLogger(__name__)


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
        self.settings = get_settings()
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

    def _get_generation_config(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> GenerationConfig:
        """Creates a GenerationConfig object."""
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

    def _prepare_tools(self, tools: Optional[List[ToolFunction]] = None) -> Optional[List[Tool | dict]]:
        """Converts internal ToolFunction list to Gemini API Tool format."""
        if not tools or (FunctionDeclaration is dict or Tool is dict):
            return None # Return None if tools aren't provided or types couldn't be imported

        gemini_tools = []
        try:
            for tool_func in tools:
                # Convert Pydantic parameters to OpenAPI JSON Schema format
                properties = {}
                required = []
                for param in tool_func.parameters:
                    param_schema = {
                        "type": param.type,
                        "description": param.description
                    }
                    if param.enum:
                        param_schema["enum"] = param.enum
                    properties[param.name] = param_schema
                    if param.required:
                        required.append(param.name)

                openapi_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }

                function_decl = FunctionDeclaration(
                    name=tool_func.name,
                    description=tool_func.description,
                    parameters=openapi_schema
                )
                gemini_tools.append(Tool(function_declarations=[function_decl])) # Gemini expects a list

            return gemini_tools
        except Exception as e:
            logger.error(f"Failed to prepare tools for Gemini API: {e}", exc_info=True)
            return None # Return None if preparation fails

    async def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[ToolFunction]] = None
    ) -> Tuple[str, Dict[str, int], Optional[List[Dict]]]:
        """
        Generate text using the configured Gemini model.

        Args:
            prompt: The input text prompt.
            temperature: Override generation temperature.
            max_tokens: Override maximum tokens to generate.
            tools: Optional list of tools for function calling.

        Returns:
            Tuple containing:
                - Generated text (str).
                - Token usage info (Dict[str, int]).
                - Optional list of tool calls requested by the model (List[Dict]).

        Raises:
            LLMServiceError: If the API call fails.
            RateLimitError: If rate limits are exceeded.
            AuthenticationError: If authentication fails.
            ConfigurationError: If the service is not configured.
        """
        if self.use_mock:
            return self._mock_generate_text(prompt, tools)
        if not self.client:
             raise ConfigurationError("Gemini client is not initialized. Check API key and configuration.")

        generation_config = self._get_generation_config(temperature, max_tokens)
        gemini_tools = self._prepare_tools(tools)

        logger.debug(f"Generating text with model {self.model_name}. Prompt length: {len(prompt)}")
        try:
            response = await asyncio.to_thread(
                self.client.generate_content,
                contents=[prompt], # Pass prompt as contents list
                generation_config=generation_config,
                tools=gemini_tools if gemini_tools else None
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
                        # Find the original ToolFunction definition (needed for parameters etc.)
                        matched_tool_func = next((t for t in tools if t.name == fc.name), None) if tools else None
                        if matched_tool_func:
                             # Ensure args are a dictionary
                            try:
                                args_dict = dict(fc.args)
                            except Exception:
                                logger.error(f"Could not convert function call args to dict for {fc.name}. Args: {fc.args}")
                                args_dict = {}

                            tool_calls_list.append({
                                "name": fc.name,
                                "args": args_dict,
                                # Include full function definition if needed by caller
                                # "function_definition": matched_tool_func.dict()
                            })
                        else:
                            logger.warning(f"Model requested unknown function: {fc.name}")

            # --- Token Usage ---
            # Accessing usage metadata might vary slightly based on API version/response structure
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

        # --- Error Handling ---
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
        tools: Optional[List[ToolFunction]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream conversation response chunks from Gemini.

        Yields dictionaries containing 'text' for text chunks or 'function_call'
        for function call requests.

        Args:
            context: Full conversation context string.
            temperature: Override generation temperature.
            max_tokens: Override maximum tokens to generate.
            tools: Optional list of tools for function calling.

        Yields:
            Dict[str, Any]: Chunks of the response, e.g., {"text": "..."} or
                            {"function_call": {"name": "...", "args": {...}}}

        Raises:
            LLMServiceError: If the API call fails.
            RateLimitError: If rate limits are exceeded.
            AuthenticationError: If authentication fails.
            ConfigurationError: If the service is not configured.
        """
        if self.use_mock:
            async for chunk in self._mock_stream_conversation(context):
                yield chunk
            return
        if not self.client:
            raise ConfigurationError("Gemini client is not initialized. Check API key and configuration.")

        generation_config = self._get_generation_config(temperature, max_tokens)
        gemini_tools = self._prepare_tools(tools)

        logger.debug(f"Streaming conversation with model {self.model_name}. Context length: {len(context)}")
        try:
            # Start the streaming generation
            stream = await asyncio.to_thread(
                self.client.generate_content,
                contents=[context],
                generation_config=generation_config,
                tools=gemini_tools if gemini_tools else None,
                stream=True
            )

            # Process the stream
            async for chunk in stream:
                if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'text') and part.text:
                            yield {"text": part.text}
                        elif hasattr(part, 'function_call'):
                            fc = part.function_call
                            try:
                                args_dict = dict(fc.args)
                            except Exception:
                                logger.error(f"Could not convert streaming function call args to dict for {fc.name}. Args: {fc.args}")
                                args_dict = {}
                            yield {"function_call": {"name": fc.name, "args": args_dict}}
                # Include usage metadata if available at the end (might be in the last chunk or separate)
                # usage_metadata = getattr(chunk, 'usage_metadata', None)
                # if usage_metadata:
                #     yield {"usage_metadata": usage_metadata.to_dict()} # Or process as needed

        # --- Error Handling ---
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
        except Exception as e:
            logger.exception(f"Unexpected error during Gemini streaming: {e}", exc_info=True)
            raise LLMServiceError(f"Unexpected error streaming conversation: {e}")

    # --- Mock Methods ---
    def _mock_generate_text(
        self, prompt: str, tools: Optional[List[ToolFunction]] = None
    ) -> Tuple[str, Dict[str, int], Optional[List[Dict]]]:
        logger.warning("Using mock Gemini text generation")
        response_text = f"Mock response for prompt: '{prompt[:50]}...'"
        token_info = {"input_tokens": len(prompt)//4, "output_tokens": 20, "total_tokens": len(prompt)//4 + 20}
        tool_calls = None

        # Simple mock function call example
        if tools and "weather" in prompt.lower():
             weather_tool = next((t for t in tools if t.name == "get_weather"), None)
             if weather_tool:
                  response_text = "Okay, I can get the weather for you."
                  tool_calls = [{"name": "get_weather", "args": {"location": "San Francisco"}}]

        return response_text, token_info, tool_calls

    async def _mock_stream_conversation(self, context: str) -> AsyncGenerator[Dict[str, Any], None]:
        logger.warning("Using mock Gemini conversation streaming")
        response = f"Mock stream response for context: '{context[:30]}...'. This is chunked."
        words = response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.05) # Simulate delay
            yield {"text": word + (" " if i < len(words) - 1 else "")}
        # Optionally yield a mock function call at the end
        # yield {"function_call": {"name": "mock_function", "args": {"param": "value"}}}


# --- Helper Functions (Potentially move to appropriate modules if needed) ---

# Caching transcription might be useful, but depends on audio data variability
# @cached(ttl=3600, key_prefix="audio_transcription") # Consider cache invalidation strategy
async def transcribe_audio(audio_data: bytes) -> str:
    """
    Transcribe audio data to text using an appropriate service (Placeholder).
    NOTE: Gemini API itself might have audio input capabilities, investigate that.
          This is a placeholder and doesn't call Gemini.
    """
    logger.info(f"Received audio data for transcription: {len(audio_data)} bytes")
    # Placeholder: Replace with actual transcription logic (e.g., Google Cloud Speech-to-Text)
    # or investigate using Gemini's multimodal capabilities if suitable.
    if not audio_data:
        return ""

    # Mock implementation:
    await asyncio.sleep(0.2) # Simulate processing time
    mock_transcription = f"Mock transcription of {len(audio_data)} bytes of audio."
    if len(audio_data) > 50000:
         mock_transcription += " It seems like a longer recording."

    logger.info(f"Mock transcription result: '{mock_transcription}'")
    return mock_transcription

# Caching voice generation might be useful
# @cached(ttl=86400, key_prefix="voice_gen") # Cache for a day
async def generate_voice_response(text: str) -> bytes:
    """
    Generate voice (audio bytes) from text using an appropriate service (Placeholder).
    NOTE: This is a placeholder and doesn't call Gemini or a TTS service.
    """
    logger.info(f"Generating mock voice response for text: '{text[:50]}...'")
    # Placeholder: Replace with actual TTS logic (e.g., Google Cloud Text-to-Speech)
    await asyncio.sleep(0.1) # Simulate processing time
    # Create simple mock audio data (e.g., repeating the text)
    mock_audio = f"AUDIO_FOR:[{text}]".encode('utf-8') * 3 # Repeat to make it seem like audio
    # Return raw bytes, base64 encoding should happen closer to the WebSocket send if needed
    return mock_audio