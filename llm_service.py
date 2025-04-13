"""
LLM Service module for EVA backend.

This module provides enhanced integration with Gemini API for text generation,
streaming responses, and function calling capabilities with memory integration.

Update your existing llm_service.py file with this version.

Current Date: 2025-04-13
Current User: IAmLep
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
from google.api_core.exceptions import GoogleAPIError
from pydantic import BaseModel, Field
from fastapi import HTTPException, Request, Response, status

# Update imports for compatibility with latest google-generativeai SDK
try:
    # First try the newer import path (0.3.0+)
    from google.generativeai.types import FunctionDeclaration, Tool
    from google.generativeai.types.generation_types import GenerationConfig
except ImportError:
    try:
        # Fall back to alternative import paths
        from google.generativeai.types.generation_types import FunctionDeclaration, Tool, GenerationConfig
    except ImportError:
        # If both fail, use direct dictionary objects instead of these classes
        logging.warning("Could not import Tool types, using dict-based format instead")
        # Define placeholder classes that we'll use in the code
        class FunctionDeclaration(dict):
            def __init__(self, name, description, parameters):
                super().__init__(name=name, description=description, parameters=parameters)
                
        class Tool(dict):
            def __init__(self, function_declarations):
                super().__init__(function_declarations=function_declarations)
        
        # Also create a placeholder for GenerationConfig
        class GenerationConfig(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

from cache_manager import cached
from config import get_settings
from exceptions import LLMServiceError, RateLimitError
from models import Memory

# Logger configuration
logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """
    Tool parameter model for function calling.
    """
    name: str
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


class ToolFunction(BaseModel):
    """
    Tool function model for Gemini API function calling.
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    response_description: Optional[str] = None


class ToolCall(BaseModel):
    """
    Tool call model representing a function called by the LLM.
    """
    function: ToolFunction
    args: Dict[str, Any]


class StreamProgress(BaseModel):
    """
    Streaming progress model for tracking streaming responses.
    """
    content: str = ""
    is_complete: bool = False
    tool_calls: List[ToolCall] = Field(default_factory=list)
    token_count: int = 0


class GeminiService:
    """
    Enhanced service for interacting with Google's Gemini API.
    
    This class provides methods for generating text responses,
    streaming conversations, and handling function calls.
    """
    
    def __init__(self):
        """Initialize Gemini service with API key from settings."""
        self.settings = get_settings()
        self.api_key = self.settings.GEMINI_API_KEY
        
        if not self.api_key:
            logger.warning("No Gemini API key found, using mock responses")
            self.use_mock = True
        else:
            self.use_mock = False
            genai.configure(api_key=self.api_key)
        
        logger.info("Gemini service initialized")
    
    async def generate_text(
        self, 
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[ToolFunction]] = None
    ) -> Tuple[str, Dict[str, int], Optional[List[ToolCall]]]:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: Text prompt
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools for function calling
            
        Returns:
            Tuple[str, Dict[str, int], Optional[List[ToolCall]]]: 
                Generated text, token info, and optional tool calls
        """
        if self.use_mock:
            return self._mock_generate_text(prompt, tools)
        
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40,
            )
            
            # Configure model
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            # Handle function calling if tools are provided
            if tools:
                gemini_tools = []
                
                for tool in tools:
                    # Convert Pydantic model to Gemini API format
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    for param in tool.parameters:
                        parameters["properties"][param.name] = {
                            "type": param.type,
                            "description": param.description
                        }
                        
                        if param.enum:
                            parameters["properties"][param.name]["enum"] = param.enum
                        
                        if param.required:
                            parameters["required"].append(param.name)
                    
                    function = FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=parameters
                    )
                    
                    gemini_tools.append(Tool(function=function))
                
                # Set tools for the model
                model.tools = gemini_tools
            
            # Generate response
            response = await asyncio.to_thread(
                model.generate_content,
                prompt
            )
            
            # Extract token counts
            token_info = {
                "input_tokens": getattr(response, "prompt_token_count", 0),
                "output_tokens": getattr(response, "candidates_token_count", 0),
                "total_tokens": getattr(response, "total_token_count", 0)
            }
            
            # Extract tool calls
            tool_calls = []
            if tools and hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                function_call = part.function_call
                                
                                # Find matching tool
                                matched_tool = next(
                                    (t for t in tools if t.name == function_call.name),
                                    None
                                )
                                
                                if matched_tool:
                                    # Parse arguments
                                    args = json.loads(
                                        function_call.args 
                                        if isinstance(function_call.args, str) 
                                        else json.dumps(function_call.args)
                                    )
                                    
                                    tool_calls.append(ToolCall(
                                        function=matched_tool,
                                        args=args
                                    ))
            
            return response.text, token_info, tool_calls if tool_calls else None
        
        except Exception as e:
            error_message = str(e).lower()
            
            if "rate limit" in error_message or "quota" in error_message:
                logger.warning(f"Gemini API rate limit exceeded: {str(e)}")
                raise RateLimitError("Gemini API rate limit exceeded")
            
            logger.error(f"Gemini API error: {str(e)}")
            raise LLMServiceError(f"Failed to generate text: {str(e)}")
    
    async def stream_conversation(
        self,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[ToolFunction]] = None,
        callback: Optional[Callable[[str, bool], None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream conversation response from Gemini.
        
        Args:
            context: Full context including system prompt, memories, and conversation
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools for function calling
            callback: Optional callback function for each chunk
            
        Yields:
            str: Response chunks as they are generated
        """
        if self.use_mock:
            async for chunk in self._mock_stream_conversation(context):
                if callback:
                    callback(chunk, False)
                yield chunk
            return
            
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=0.95,
                top_k=40,
            )
            
            # Configure model
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                generation_config=generation_config
            )
            
            # Handle function calling if tools are provided
            if tools:
                gemini_tools = []
                
                for tool in tools:
                    # Convert Pydantic model to Gemini API format
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                    
                    for param in tool.parameters:
                        parameters["properties"][param.name] = {
                            "type": param.type,
                            "description": param.description
                        }
                        
                        if param.enum:
                            parameters["properties"][param.name]["enum"] = param.enum
                        
                        if param.required:
                            parameters["required"].append(param.name)
                    
                    function = FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=parameters
                    )
                    
                    gemini_tools.append(Tool(function=function))
                
                # Set tools for the model
                model.tools = gemini_tools
            
            # Generate streaming response
            stream_response = await asyncio.to_thread(
                model.generate_content,
                context,
                stream=True
            )
            
            # Process streaming chunks
            full_response = ""
            
            for chunk in stream_response:
                if hasattr(chunk, 'text') and chunk.text:
                    chunk_text = chunk.text
                    full_response += chunk_text
                    
                    if callback:
                        callback(chunk_text, False)
                    
                    yield chunk_text
            
            # Final callback with complete flag
            if callback:
                callback(full_response, True)
            
        except Exception as e:
            error_message = str(e).lower()
            
            if "rate limit" in error_message or "quota" in error_message:
                logger.warning(f"Gemini API rate limit exceeded: {str(e)}")
                raise RateLimitError("Gemini API rate limit exceeded")
            
            logger.error(f"Gemini API streaming error: {str(e)}")
            raise LLMServiceError(f"Failed to stream conversation: {str(e)}")
    
    async def process_conversation_with_context(
        self, 
        context_text: str,
        is_streaming: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: Optional[List[ToolFunction]] = None,
        callback: Optional[Callable[[str, bool], None]] = None
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Process a conversation with full context.
        
        Args:
            context_text: Full context text to send to Gemini
            is_streaming: Whether to use streaming response
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools for function calling
            callback: Optional callback function for streaming chunks
            
        Returns:
            Union[str, AsyncGenerator[str, None]]: Generated response or stream
        """
        if is_streaming:
            return self.stream_conversation(
                context_text,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                callback=callback
            )
        else:
            response, _, _ = await self.generate_text(
                context_text,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools
            )
            return response
    
    async def _mock_stream_conversation(self, context: str) -> AsyncGenerator[str, None]:
        """
        Generate mock streaming responses for testing.
        
        Args:
            context: Context text (ignored in mock)
            
        Yields:
            str: Mock response chunks
        """
        logger.warning("Using mock conversation streaming")
        
        # Create a mock response that streams in chunks
        response = "Hello! I'm your AI assistant called EVA. I'm here to help you with whatever you need. How can I assist you today?"
        chunks = [response[i:i+10] for i in range(0, len(response), 10)]
        
        for chunk in chunks:
            await asyncio.sleep(0.1)  # Simulate network delay
            yield chunk
    
    def _mock_generate_text(
        self, 
        prompt: str,
        tools: Optional[List[ToolFunction]] = None
    ) -> Tuple[str, Dict[str, int], Optional[List[ToolCall]]]:
        """
        Generate mock text responses for testing.
        
        Args:
            prompt: Text prompt
            tools: Optional list of tools for function calling
            
        Returns:
            Tuple[str, Dict[str, int], Optional[List[ToolCall]]]: 
                Mock generated text, token counts, and optional tool calls
        """
        logger.warning("Using mock text generation")
        
        # Generate a simple response based on the prompt
        if "weather" in prompt.lower():
            response = "The weather is currently sunny with a temperature of 25°C."
            
            # If weather tool is available, create a tool call
            if tools and any(t.name == "get_weather" for t in tools):
                weather_tool = next(t for t in tools if t.name == "get_weather")
                tool_calls = [
                    ToolCall(
                        function=weather_tool,
                        args={"location": "Current location", "unit": "celsius"}
                    )
                ]
                return response, {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}, tool_calls
        
        elif "time" in prompt.lower():
            current_time = datetime.utcnow().strftime("%H:%M:%S")
            response = f"The current time is {current_time} UTC."
        
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            response = "Hello! How can I assist you today?"
        
        else:
            response = "I understand you're asking about something. How can I help you with that?"
        
        # Return mock token count
        token_info = {"input_tokens": 50, "output_tokens": 20, "total_tokens": 70}
        
        return response, token_info, None


async def generate_response(
    context_text: str,
    is_streaming: bool = False,
    tools: Optional[List[ToolFunction]] = None,
    callback: Optional[Callable[[str, bool], None]] = None
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generate a response using the context window.
    
    Args:
        context_text: Full context text
        is_streaming: Whether to use streaming response
        tools: Optional list of tools for function calling
        callback: Optional callback for streaming chunks
        
    Returns:
        Union[str, AsyncGenerator[str, None]]: Response or stream
    """
    service = GeminiService()
    return await service.process_conversation_with_context(
        context_text,
        is_streaming=is_streaming,
        tools=tools,
        callback=callback
    )


# We keep the existing transcription functions for compatibility
@cached(ttl=3600, key_prefix="audio_transcription")
async def transcribe_audio(audio_data: bytes) -> Optional[str]:
    """Transcribe audio data to text."""
    # Implementation kept from original file
    try:
        service = GeminiService()
        
        if service.use_mock:
            return "This is a mock transcription of audio data."
            
        # Placeholder for actual implementation
        await asyncio.sleep(0.5)
        
        audio_length = len(audio_data)
        if audio_length < 1000:
            return "Hello, this is a short audio message."
        elif audio_length < 10000:
            return "This is a medium length message about using the voice interface."
        else:
            return "This is a longer message that demonstrates how the transcription service works."
    
    except Exception as e:
        logger.error(f"Audio transcription error: {str(e)}")
        raise LLMServiceError(f"Failed to transcribe audio: {str(e)}")


async def process_audio_stream(audio_data: bytes) -> str:
    """Process audio stream data and transcribe to text."""
    try:
        transcription = await transcribe_audio(audio_data)
        if not transcription:
            raise LLMServiceError("Failed to transcribe audio: Empty result")
        
        logger.info(f"Successfully processed audio stream of {len(audio_data)} bytes")
        return transcription
    
    except Exception as e:
        logger.error(f"Error processing audio stream: {str(e)}")
        raise LLMServiceError(f"Failed to process audio stream: {str(e)}")


@cached(ttl=3600, key_prefix="voice_generation")
async def generate_voice_response(text: str) -> bytes:
    """Generate voice response from text."""
    # Implementation kept from original file
    try:
        service = GeminiService()
        
        if service.use_mock:
            mock_audio = b"MOCK_AUDIO_DATA" * 100
            return base64.b64encode(mock_audio)
            
        logger.info(f"Generating voice for text of length {len(text)}")
        
        await asyncio.sleep(0.5)
        
        mock_audio = b"AUDIO_DATA_FOR:" + text.encode('utf-8')
        return base64.b64encode(mock_audio)
    
    except Exception as e:
        logger.error(f"Voice generation error: {str(e)}")
        raise LLMServiceError(f"Failed to generate voice: {str(e)}")