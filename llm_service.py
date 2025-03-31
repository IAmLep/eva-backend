import os
import logging
import google.generativeai as genai
from typing import Any, Dict, List, Optional, Callable, Awaitable
import json
import asyncio

logger = logging.getLogger(__name__)

class LLMService:
    """Service for interacting with Gemini API"""
    
    def __init__(self):
        # Set up Gemini API key
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not set in environment variables")
            raise ValueError("GEMINI_API_KEY not set")
            
        genai.configure(api_key=api_key)
        
        # Get the model (Gemini 1.5 Flash)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
    
    async def generate_text(self, prompt: str, memory: str = "", max_tokens: int = 1024) -> str:
        """Generate text with Gemini model"""
        try:
            # Include memory in the prompt if available
            full_prompt = prompt
            if memory:
                full_prompt = f"MEMORY CONTEXT:\n{memory}\n\nUSER QUERY:\n{prompt}"
            
            # Generate the response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config={"max_output_tokens": max_tokens}
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    async def generate_text_streaming(
        self, 
        prompt: str, 
        memory: str = "", 
        callback: Optional[Callable[[str], Awaitable[None]]] = None
    ) -> str:
        """Generate text with streaming responses"""
        try:
            # Include memory in the prompt if available
            full_prompt = prompt
            if memory:
                full_prompt = f"MEMORY CONTEXT:\n{memory}\n\nUSER QUERY:\n{prompt}"
            
            # Generate the response with streaming
            stream = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                stream=True
            )
            
            full_response = ""
            
            # Process the streaming response
            async for chunk in self._process_stream(stream):
                full_response += chunk
                if callback:
                    await callback(chunk)
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in streaming text generation: {str(e)}")
            error_msg = f"I'm sorry, I encountered an error: {str(e)}"
            if callback:
                await callback(error_msg)
            return error_msg
    
    async def _process_stream(self, stream):
        """Process the streaming response from Gemini"""
        for chunk in stream:
            if chunk.text:
                yield chunk.text
            await asyncio.sleep(0)  # Allow other tasks to run
    
    async def generate_with_function_calling(self, prompt: str, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate text with function calling capabilities"""
        try:
            # Configure the model with tools
            model_with_tools = self.model.with_tools(tools)
            
            # Generate response
            response = await asyncio.to_thread(
                model_with_tools.generate_content,
                prompt
            )
            
            # Check if the response contains a function call
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            # Return the function call information
                            return {
                                "type": "function_call",
                                "function_name": part.function_call.name,
                                "arguments": json.loads(part.function_call.args)
                            }
            
            # If no function call, return the text response
            return {
                "type": "text",
                "content": response.text
            }
            
        except Exception as e:
            logger.error(f"Error in function calling: {str(e)}")
            return {
                "type": "error",
                "error": str(e)
            }