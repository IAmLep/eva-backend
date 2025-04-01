"""
LLM Service module for EVA backend.

This module provides integration with Gemini API for text generation,
transcription, and function calling capabilities.

Last updated: 2025-04-01 10:17:14
Version: v1.8.7
Created by: IAmLep
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
from fastapi import HTTPException, status

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
    
    Attributes:
        name: Parameter name
        type: Parameter type
        description: Parameter description
        required: Whether parameter is required
        enum: Optional list of allowed values
    """
    name: str
    type: str
    description: str
    required: bool = False
    enum: Optional[List[str]] = None


class ToolFunction(BaseModel):
    """
    Tool function model for Gemini API function calling.
    
    Attributes:
        name: Function name
        description: Function description
        parameters: List of function parameters
        response_description: Description of function response
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    response_description: Optional[str] = None


class ToolCall(BaseModel):
    """
    Tool call model representing a function called by the LLM.
    
    Attributes:
        function: Function that was called
        args: Arguments passed to the function
    """
    function: ToolFunction
    args: Dict[str, Any]


class GeminiService:
    """
    Service for interacting with Google's Gemini API.
    
    This class provides methods for generating text responses,
    processing conversations with function calling, and handling
    voice transcription and generation.
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
    ) -> Tuple[str, Optional[List[ToolCall]]]:
        """
        Generate text using Gemini API.
        
        Args:
            prompt: Text prompt
            temperature: Generation temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            tools: Optional list of tools for function calling
            
        Returns:
            Tuple[str, Optional[List[ToolCall]]]: 
                Generated text and optional tool calls
                
        Raises:
            LLMServiceError: If generation fails
            RateLimitError: If rate limit is exceeded
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
                model_name="gemini-1.5-pro",
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
                
                # Generate response with tools
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )
                
                # Extract tool calls
                tool_calls = []
                if hasattr(response, 'candidates') and response.candidates:
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
                
                # Get text response
                text_response = response.text
                return text_response, tool_calls if tool_calls else None
            else:
                # Generate response without tools
                response = await asyncio.to_thread(
                    model.generate_content,
                    prompt
                )
                
                return response.text, None
        
        except Exception as e:
            error_message = str(e).lower()
            
            if "rate limit" in error_message or "quota" in error_message:
                logger.warning(f"Gemini API rate limit exceeded: {str(e)}")
                raise RateLimitError("Gemini API rate limit exceeded")
            
            logger.error(f"Gemini API error: {str(e)}")
            raise LLMServiceError(f"Failed to generate text: {str(e)}")
    
    async def process_conversation(
        self, 
        message: str, 
        user_id: str, 
        conversation_id: Optional[str] = None,
        memories: Optional[List[Memory]] = None,
        tools: Optional[List[ToolFunction]] = None
    ) -> Tuple[str, str, Optional[List[ToolCall]]]:
        """
        Process a conversation message with context.
        
        Args:
            message: User message
            user_id: User ID
            conversation_id: Optional conversation ID
            memories: Optional list of memories for context
            tools: Optional list of tools for function calling
            
        Returns:
            Tuple[str, str, Optional[List[ToolCall]]]:
                Generated response, conversation ID, and optional tool calls
        """
        # Generate or use existing conversation ID
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        # Build context with memories
        context = ""
        if memories:
            # Only use a limited amount of memory content to avoid token limits
            memory_text = "\n".join(
                f"Memory: {m.content[:500]}..." if len(m.content) > 500 else f"Memory: {m.content}"
                for m in memories[:5]
            )
            context = f"User memories:\n{memory_text}\n\n"
        
        # Format system prompt with context
        system_prompt = f"""
You are EVA, a helpful AI assistant. 
Current date and time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC.

{context}

Respond to the user's message conversationally and helpfully. 
If the user asks you to perform a specific function, use the provided function calling tools.
Always provide accurate and useful information.
"""
        
        # Combine prompts
        full_prompt = f"{system_prompt}\n\nUser: {message}\nEVA:"
        
        # Generate response
        response, tool_calls = await self.generate_text(
            full_prompt,
            temperature=0.85,
            max_tokens=1024,
            tools=tools
        )
        
        logger.info(f"Generated response for conversation {conversation_id[:8]}")
        return response, conversation_id, tool_calls
    
    def _mock_generate_text(
        self, 
        prompt: str,
        tools: Optional[List[ToolFunction]] = None
    ) -> Tuple[str, Optional[List[ToolCall]]]:
        """
        Generate mock text responses for testing without API key.
        
        Args:
            prompt: Text prompt
            tools: Optional list of tools for function calling
            
        Returns:
            Tuple[str, Optional[List[ToolCall]]]:
                Mock generated text and optional tool calls
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
                return response, tool_calls
        
        elif "time" in prompt.lower():
            current_time = datetime.utcnow().strftime("%H:%M:%S")
            response = f"The current time is {current_time} UTC."
        
        elif "hello" in prompt.lower() or "hi" in prompt.lower():
            response = "Hello! How can I assist you today?"
        
        else:
            response = "I understand you're asking about something. How can I help you with that?"
        
        return response, None


async def generate_response(
    message: str, 
    user_id: str, 
    conversation_id: Optional[str] = None,
    memories: Optional[List[Memory]] = None,
    tools: Optional[List[ToolFunction]] = None
) -> Tuple[str, str, Optional[List[ToolCall]]]:
    """
    Generate a response to a user message.
    
    Args:
        message: User message
        user_id: User ID
        conversation_id: Optional conversation ID
        memories: Optional list of memories for context
        tools: Optional list of tools for function calling
        
    Returns:
        Tuple[str, str, Optional[List[ToolCall]]]:
            Generated response, conversation ID, and optional tool calls
    """
    service = GeminiService()
    return await service.process_conversation(
        message, 
        user_id, 
        conversation_id,
        memories,
        tools
    )


@cached(ttl=3600, key_prefix="audio_transcription")
async def transcribe_audio(audio_data: bytes) -> Optional[str]:
    """
    Transcribe audio data to text.
    
    Args:
        audio_data: Audio data as bytes
        
    Returns:
        Optional[str]: Transcribed text or None if failed
        
    Raises:
        LLMServiceError: If transcription fails
    """
    # In a real implementation, this would call a speech-to-text API
    # For this refactoring, we'll mock it with a simple function
    try:
        service = GeminiService()
        
        if service.use_mock:
            # Return mock transcription
            return "This is a mock transcription of audio data."
        
        # Using Google's Speech-to-Text would be implemented here
        # For now, we'll use a placeholder that could be replaced with actual implementation
        logger.info(f"Transcribing {len(audio_data)} bytes of audio data")
        
        # This is a placeholder for the actual implementation
        # In practice, you would use Google's Speech-to-Text API or similar
        await asyncio.sleep(0.5)  # Simulate API call
        
        # Mock successful transcription with the audio length affecting the response
        audio_length = len(audio_data)
        if audio_length < 1000:
            return "Hello, this is a short audio message."
        elif audio_length < 10000:
            return "This is a medium length message about using the voice interface."
        else:
            return "This is a longer message that demonstrates how the transcription service works with more substantial audio content. The user might be asking about specific features or giving detailed instructions."
    
    except Exception as e:
        logger.error(f"Audio transcription error: {str(e)}")
        raise LLMServiceError(f"Failed to transcribe audio: {str(e)}")


@cached(ttl=3600, key_prefix="voice_generation")
async def generate_voice_response(text: str) -> bytes:
    """
    Generate voice response from text.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        bytes: Audio data as bytes
        
    Raises:
        LLMServiceError: If generation fails
    """
    # In a real implementation, this would call a text-to-speech API
    # For this refactoring, we'll mock it with a simple function
    try:
        service = GeminiService()
        
        if service.use_mock:
            # Return mock audio data
            mock_audio = b"MOCK_AUDIO_DATA" * 100
            return base64.b64encode(mock_audio)
        
        # Using Google's Text-to-Speech would be implemented here
        # For now, we'll use a placeholder that could be replaced with actual implementation
        logger.info(f"Generating voice for text of length {len(text)}")
        
        # This is a placeholder for the actual implementation
        # In practice, you would use Google's Text-to-Speech API or similar
        await asyncio.sleep(0.5)  # Simulate API call
        
        # Mock successful generation
        mock_audio = b"AUDIO_DATA_FOR:" + text.encode('utf-8')
        return base64.b64encode(mock_audio)
    
    except Exception as e:
        logger.error(f"Voice generation error: {str(e)}")
        raise LLMServiceError(f"Failed to generate voice: {str(e)}")