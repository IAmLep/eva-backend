import os
import json
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Callable

# Configure API clients
genai.configure(api_key=GEMINI_API_KEY)

# Default model
DEFAULT_MODEL = GEMINI_MODEL

def generate_response(conversation_history, model=GEMINI_MODEL):
    """Generate a complete response using Gemini"""
    try:
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Format conversation for Gemini
        formatted_history = []
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})
        
        convo = model_instance.start_chat(history=formatted_history[:-1])
        response = convo.send_message(conversation_history[-1]["content"])
        
        return response.text
    
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again later."

async def generate_streaming_response(
    conversation_history: List[Dict[str, str]], 
    model: str = GEMINI_MODEL,
    callback: Callable[[str], Any] = None
) -> AsyncGenerator[str, None]:
    """Stream a response using Gemini"""
    try:
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Format conversation for Gemini
        formatted_history = []
        for msg in conversation_history:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({"role": role, "parts": [msg["content"]]})
        
        convo = model_instance.start_chat(history=formatted_history[:-1])
        
        response_generator = convo.send_message_streaming(conversation_history[-1]["content"])
        
        for chunk in response_generator:
            # Extract text from chunk
            if chunk.text:
                if callback:
                    await callback(chunk.text)
                yield chunk.text
            # Small delay to allow other async operations
            await asyncio.sleep(0.01)
            
    except Exception as e:
        error_msg = f"Error streaming Gemini response: {e}"
        print(error_msg)
        if callback:
            await callback("I apologize, but I encountered an error processing your request.")
        yield "I apologize, but I encountered an error while processing your request. Please try again later."