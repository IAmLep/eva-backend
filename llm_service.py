import os
import google.generativeai as genai
import logging
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
import asyncio
from logging_config import setup_logging

# Setup logging
logger = logging.getLogger(__name__)

class LLMService:
    """Service to interact with Google's Generative AI models"""
    
    def __init__(self):
        # Configure API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not set")
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Model configuration
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"LLM Service initialized with model: {self.model_name}")
    
    async def generate_text(self, prompt: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate text response from the model"""
        try:
            # Convert chat history to the format expected by the model
            messages = []
            if chat_history:
                for msg in chat_history:
                    role = "user" if msg.get("role") == "user" else "model"
                    messages.append({"role": role, "parts": [msg.get("content", "")]})
            
            # Add the current prompt
            messages.append({"role": "user", "parts": [prompt]})
            
            # Create a chat session if needed
            chat_session = self.model.start_chat(history=messages[:-1] if messages else [])
            
            # Generate the response
            response = chat_session.send_message(prompt)
            
            logger.info(f"Generated response for prompt: {prompt[:50]}...")
            return response.text
            
        except genai.types.BlockedPromptException as e:
            logger.error(f"Blocked prompt error: {str(e)}")
            return f"I'm sorry, but I cannot respond to this prompt because it violates content safety guidelines."
        
        except genai.types.GenerateContentResponseError as e:
            logger.error(f"Error generating content: {str(e)}")
            return f"I'm sorry, I encountered an error generating content: {str(e)}"
        
        except genai.types.LimitExceededException as e:
            logger.error(f"API limit exceeded: {str(e)}")
            return f"I'm sorry, the service is currently at capacity. Please try again later."
        
        except ConnectionError as e:
            logger.error(f"Connection error: {str(e)}")
            return f"I'm sorry, there was a problem connecting to the AI service. Please check your internet connection."
        
        except Exception as e:
            logger.error(f"Unexpected error in generate_text: {str(e)}")
            return f"I'm sorry, I encountered an unexpected error. Please try again later."
    
    async def generate_text_streaming(
        self, 
        prompt: str, 
        callback: Callable[[str], None],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate text response from the model with streaming"""
        try:
            # Convert chat history to the format expected by the model
            messages = []
            if chat_history:
                for msg in chat_history:
                    role = "user" if msg.get("role") == "user" else "model"
                    messages.append({"role": role, "parts": [msg.get("content", "")]})
            
            # Create a chat session if needed
            chat_session = self.model.start_chat(history=messages if messages else [])
            
            # Generate the response with streaming
            response = chat_session.send_message(prompt, stream=True)
            
            logger.info(f"Streaming response for prompt: {prompt[:50]}...")
            
            # Stream the response chunks
            full_response = ""
            async for chunk in response:
                try:
                    chunk_text = chunk.text
                    full_response += chunk_text
                    callback(chunk_text)
                    yield chunk_text
                except Exception as e:
                    logger.error(f"Error in streaming callback: {str(e)}")
                    # Continue streaming even if callback fails
            
            return full_response
            
        except genai.types.BlockedPromptException as e:
            error_msg = f"I'm sorry, but I cannot respond to this prompt because it violates content safety guidelines."
            logger.error(f"Blocked prompt error: {str(e)}")
            callback(error_msg)
            yield error_msg
        
        except genai.types.GenerateContentResponseError as e:
            error_msg = f"I'm sorry, I encountered an error generating content: {str(e)}"
            logger.error(f"Error generating content: {str(e)}")
            callback(error_msg)
            yield error_msg
        
        except genai.types.LimitExceededException as e:
            error_msg = f"I'm sorry, the service is currently at capacity. Please try again later."
            logger.error(f"API limit exceeded: {str(e)}")
            callback(error_msg)
            yield error_msg
        
        except ConnectionError as e:
            error_msg = f"I'm sorry, there was a problem connecting to the AI service. Please check your internet connection."
            logger.error(f"Connection error: {str(e)}")
            callback(error_msg)
            yield error_msg
        
        except Exception as e:
            error_msg = f"I'm sorry, I encountered an unexpected error. Please try again later."
            logger.error(f"Unexpected error in generate_text_streaming: {str(e)}")
            callback(error_msg)
            yield error_msg
    
    async def generate_with_function_calling(
        self, 
        prompt: str, 
        tools: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate a response that may include a function call"""
        try:
            # Convert chat history to the format expected by the model
            messages = []
            if chat_history:
                for msg in chat_history:
                    role = "user" if msg.get("role") == "user" else "model"
                    messages.append({"role": role, "parts": [msg.get("content", "")]})
            
            # Add the current prompt
            messages.append({"role": "user", "parts": [prompt]})
            
            # Configure the model with tools
            model_with_tools = genai.GenerativeModel(
                model_name=self.model_name,
                tools=tools
            )
            
            # Create a chat session if needed
            chat_session = model_with_tools.start_chat(history=messages[:-1] if messages else [])
            
            # Generate the response
            response = chat_session.send_message(prompt)
            
            logger.info(f"Generated response with function calling for prompt: {prompt[:50]}...")
            
            # Check if the response contains a function call
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            return {
                                "type": "function_call",
                                "name": function_call.name,
                                "args": function_call.args,
                                "response_text": response.text
                            }
            
            # If no function call, return the text response
            return {
                "type": "text",
                "response_text": response.text
            }
            
        except genai.types.BlockedPromptException as e:
            logger.error(f"Blocked prompt error in function calling: {str(e)}")
            return {
                "type": "error",
                "error_type": "blocked_prompt",
                "response_text": f"I'm sorry, but I cannot respond to this prompt because it violates content safety guidelines."
            }
        
        except genai.types.GenerateContentResponseError as e:
            logger.error(f"Error generating content in function calling: {str(e)}")
            return {
                "type": "error",
                "error_type": "content_generation",
                "response_text": f"I'm sorry, I encountered an error generating content: {str(e)}"
            }
        
        except genai.types.LimitExceededException as e:
            logger.error(f"API limit exceeded in function calling: {str(e)}")
            return {
                "type": "error",
                "error_type": "limit_exceeded",
                "response_text": f"I'm sorry, the service is currently at capacity. Please try again later."
            }
        
        except ConnectionError as e:
            logger.error(f"Connection error in function calling: {str(e)}")
            return {
                "type": "error",
                "error_type": "connection_error",
                "response_text": f"I'm sorry, there was a problem connecting to the AI service. Please check your internet connection."
            }
        
        except Exception as e:
            logger.error(f"Unexpected error in generate_with_function_calling: {str(e)}")
            return {
                "type": "error",
                "error_type": "unexpected",
                "response_text": f"I'm sorry, I encountered an unexpected error. Please try again later."
            }

# Create a singleton instance
llm_service = LLMService()