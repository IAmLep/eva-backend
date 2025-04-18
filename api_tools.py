"""
API Tools module for EVA backend.

This module provides tools and utilities for function calling
integration with Gemini API, allowing LLM to execute actions.
Includes implementation for basic tools.
"""

import inspect
import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from datetime import datetime, timezone, timedelta # Import datetime objects

import httpx # For making HTTP requests (e.g., Weather API)
from pydantic import BaseModel, Field, create_model

# --- Local Imports ---
from auth import get_current_user # Needed for context in execute
from config import get_settings
from database import get_db_manager # Needed for Memory/Cleanup tools
from exceptions import FunctionCallError, DatabaseError, NotFoundException # Use custom exceptions
from models import User, Memory, MemorySource, MemoryCategory # Import necessary models
# Import MemoryManager for memory operations
from memory_manager import get_memory_manager

# Logger configuration
logger = logging.getLogger(__name__)

# --- Tool Definitions ---
# (ToolParameter, ToolDefinition remain the same)
class ToolParameter(BaseModel):
    name: str
    type: str # Should align with OpenAPI types (string, integer, boolean, number, array, object)
    description: str
    required: bool = False
    enum: Optional[List[str]] = None

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]

# Renamed to avoid Pydantic conflict if ToolCall is imported elsewhere
class ApiToolCall(BaseModel):
    """Represents the structure of a function call request from the LLM."""
    name: str # Function name
    args: Dict[str, Any] # Arguments provided by LLM

class Tool:
    """Base class for all tools that can be called by the LLM."""
    @staticmethod
    def get_definition() -> ToolDefinition:
        raise NotImplementedError("Tools must implement get_definition")

    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Executes the tool logic.
        Returns a JSON-serializable dictionary representing the result.
        This result will be sent back to the LLM.
        """
        raise NotImplementedError("Tools must implement execute")

# --- Implemented Tools ---

class TimeTool(Tool):
    @staticmethod
    def get_definition() -> ToolDefinition:
        return ToolDefinition(
            name="get_current_time", # Renamed for clarity
            description="Get the current time and date information.",
            parameters=[
                ToolParameter(name="timezone", type="string", description="Optional timezone (e.g., 'UTC', 'America/New_York'). Defaults to UTC.", required=False),
                ToolParameter(name="format", type="string", description="Optional format ('iso', 'human', 'unix'). Defaults to 'iso'.", required=False, enum=["iso", "human", "unix"])
            ]
        )

    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """Returns the current time, defaulting to UTC ISO format."""
        try:
            tz_str = args.get("timezone", "UTC")
            time_format = args.get("format", "iso")

            # Basic timezone handling (consider pytz for full support if needed)
            if tz_str.upper() == "UTC":
                now = datetime.now(timezone.utc)
            else:
                # Attempting local time - NOTE: Server's local time, not user's unless specified
                # For robust timezone handling, use pytz or user preferences
                logger.warning(f"TimeTool using server's local time for timezone '{tz_str}' unless it's UTC.")
                now = datetime.now() # Server's local time

            if time_format == "iso":
                time_str = now.isoformat()
            elif time_format == "human":
                time_str = now.strftime("%Y-%m-%d %H:%M:%S %Z%z") # Include timezone info
            elif time_format == "unix":
                time_str = str(int(now.timestamp()))
            else:
                time_str = now.isoformat() # Default to ISO

            return {"current_time": time_str, "timezone": tz_str, "format": time_format}
        except Exception as e:
            logger.error(f"Error executing TimeTool: {e}", exc_info=True)
            # Return error structure for LLM
            return {"error": f"Failed to get time: {e}"}

class WeatherTool(Tool):
    @staticmethod
    def get_definition() -> ToolDefinition:
        return ToolDefinition(
            name="get_weather",
            description="Get the current weather information for a specific location.",
            parameters=[
                ToolParameter(name="location", type="string", description="The city and state/country (e.g., 'London, UK', 'Tokyo', 'New York, US').", required=True),
                ToolParameter(name="units", type="string", description="Temperature units ('metric' for Celsius, 'imperial' for Fahrenheit). Defaults to 'metric'.", required=False, enum=["metric", "imperial"])
            ]
        )

    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """Fetches weather using an external API (mocked for now)."""
        settings = get_settings()
        location = args.get("location")
        units = args.get("units", "metric") # Default to Celsius

        if not location:
            return {"error": "Location parameter is required."}

        # --- Replace with actual API call ---
        # Example using OpenWeatherMap (requires API key)
        api_key = settings.WEATHER_API_KEY
        base_url = settings.WEATHER_API_URL or "https://api.openweathermap.org/data/2.5/weather"

        if not api_key or not base_url:
             logger.warning("Weather API key or URL not configured. Returning mock weather.")
             # Mock Response
             temp = 22 if units == "metric" else 72
             unit_symbol = "°C" if units == "metric" else "°F"
             return {
                 "location": location,
                 "temperature": f"{temp}{unit_symbol}",
                 "condition": "Sunny (Mock Data)",
                 "humidity_percent": 65,
                 "details": "Mock weather data. Configure WEATHER_API_KEY and WEATHER_API_URL for real data."
             }

        params = {
            "q": location,
            "appid": api_key,
            "units": units
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(base_url, params=params)
                response.raise_for_status() # Raise exception for bad status codes
                data = response.json()

            # Extract relevant information
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            temp = main.get("temp")
            feels_like = main.get("feels_like")
            humidity = main.get("humidity")
            description = weather.get("description")
            city = data.get("name")
            unit_symbol = "°C" if units == "metric" else "°F"

            return {
                "location": city or location,
                "temperature": f"{temp}{unit_symbol}" if temp is not None else "N/A",
                "feels_like": f"{feels_like}{unit_symbol}" if feels_like is not None else "N/A",
                "condition": description.capitalize() if description else "N/A",
                "humidity_percent": humidity if humidity is not None else "N/A",
            }
        except httpx.HTTPStatusError as e:
             error_detail = f"HTTP error fetching weather: {e.response.status_code}"
             if e.response.status_code == 404: error_detail = f"Could not find weather for location: {location}"
             if e.response.status_code == 401: error_detail = "Invalid weather API key."
             logger.error(f"{error_detail} - Response: {e.response.text}")
             return {"error": error_detail}
        except httpx.RequestError as e:
            logger.error(f"Network error fetching weather: {e}")
            return {"error": "Could not connect to weather service."}
        except Exception as e:
            logger.error(f"Error executing WeatherTool: {e}", exc_info=True)
            return {"error": f"Failed to get weather: {e}"}

class MemoryTool(Tool):
    @staticmethod
    def get_definition() -> ToolDefinition:
        return ToolDefinition(
            name="manage_memory",
            description="Save, retrieve, or delete user memories. Use 'retrieve' to find relevant information based on a query.",
            parameters=[
                ToolParameter(name="action", type="string", description="Action: 'save', 'retrieve', or 'delete'.", required=True, enum=["save", "retrieve", "delete"]),
                ToolParameter(name="content", type="string", description="Memory content to save (required for 'save').", required=False),
                ToolParameter(name="category", type="string", description="Category for 'save' (e.g., 'personal_info', 'preference', 'fact'). Defaults to 'fact'.", required=False, enum=[cat.value for cat in MemoryCategory]),
                ToolParameter(name="query", type="string", description="Search query for 'retrieve'.", required=False),
                ToolParameter(name="memory_id", type="string", description="ID of the memory to delete (required for 'delete').", required=False)
            ]
        )

    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """Manages user memories via MemoryManager."""
        action = args.get("action")
        memory_manager = get_memory_manager()

        try:
            if action == "save":
                content = args.get("content")
                if not content: return {"error": "Content is required for 'save' action."}
                category_str = args.get("category", MemoryCategory.FACT.value)
                try:
                    category = MemoryCategory(category_str)
                except ValueError:
                    return {"error": f"Invalid category '{category_str}'. Valid categories are: {[c.value for c in MemoryCategory]}"}

                # Using create_core_memory for simplicity, could add logic for event/conv types
                memory = await memory_manager.create_core_memory(
                    user_id=user.id,
                    content=content,
                    category=category
                )
                return {"status": "success", "action": "save", "memory_id": memory.memory_id, "message": "Memory saved."}

            elif action == "retrieve":
                query = args.get("query")
                if not query: return {"error": "Query is required for 'retrieve' action."}

                # Use get_relevant_memories
                relevant_memories = await memory_manager.get_relevant_memories(user_id=user.id, query=query, limit=3) # Limit results for LLM
                if not relevant_memories:
                    return {"status": "success", "action": "retrieve", "found_memories": 0, "message": "No relevant memories found."}

                results = [
                    {"memory_id": mem.memory_id, "content": mem.content, "relevance": score}
                    for mem, score in relevant_memories
                ]
                return {"status": "success", "action": "retrieve", "found_memories": len(results), "results": results}

            elif action == "delete":
                memory_id = args.get("memory_id")
                if not memory_id: return {"error": "Memory ID is required for 'delete' action."}

                success = await memory_manager.delete_memory(memory_id=memory_id, user_id=user.id)
                if success:
                    return {"status": "success", "action": "delete", "memory_id": memory_id, "message": "Memory deleted."}
                else:
                    # delete_memory raises NotFoundException if not found/authorized
                    # This path might not be reached if exceptions are handled globally
                    return {"status": "error", "action": "delete", "memory_id": memory_id, "message": "Memory not found or delete failed."}

            else:
                return {"error": f"Invalid action '{action}'. Must be 'save', 'retrieve', or 'delete'."}

        except (NotFoundException, AuthorizationError) as e:
            logger.warning(f"MemoryTool error ({action}): {e}")
            return {"error": str(e)} # Return specific error to LLM
        except Exception as e:
            logger.error(f"Error executing MemoryTool ({action}): {e}", exc_info=True)
            return {"error": f"Failed to {action} memory: {e}"}

# --- Tool Registration & Execution ---

# Register available tools explicitly
_available_tools_registry: Dict[str, Type[Tool]] = {
    tool.get_definition().name: tool for tool in [
        TimeTool,
        WeatherTool,
        MemoryTool,
        # Add SyncTool, CleanupTool implementations here if needed
    ]
}

def available_tools() -> List[ToolDefinition]:
    """Returns definitions of all registered tools."""
    return [tool.get_definition() for tool in _available_tools_registry.values()]

def get_tool_by_name(name: str) -> Optional[Type[Tool]]:
    """Gets a tool class by its registered name."""
    return _available_tools_registry.get(name)

async def execute_function_call(
    tool_call_request: Dict, # Expects {'name': str, 'args': dict}
    user: User
) -> Dict[str, Any]:
    """
    Finds and executes the requested tool based on the LLM request.

    Args:
        tool_call_request: Dictionary containing 'name' and 'args'.
        user: The current authenticated user.

    Returns:
        A dictionary containing the result of the tool execution, suitable
        for serialization and sending back to the LLM. Includes 'error' key on failure.

    Raises:
        FunctionCallError: If the tool is not found. (Specific tool errors are returned in the result dict)
    """
    function_name = tool_call_request.get("name")
    args = tool_call_request.get("args", {})

    if not function_name:
        logger.error("execute_function_call received request without 'name'.")
        # Don't raise, return error dict for LLM
        return {"error": "Function call request missing function name."}

    tool_class = get_tool_by_name(function_name)
    if not tool_class:
        logger.error(f"Unknown function requested by LLM: {function_name}")
        # Raise FunctionCallError here as it's a fundamental issue
        raise FunctionCallError(f"Unknown function name: {function_name}")

    logger.info(f"Executing tool '{function_name}' for user {user.id} with args: {args}")
    try:
        # Execute the tool's static method
        result = await tool_class.execute(args, user)
        # Ensure result is a dictionary
        return result if isinstance(result, dict) else {"result": str(result)}
    except Exception as e:
        # Catch unexpected errors during tool execution itself
        logger.exception(f"Unexpected error executing tool '{function_name}': {e}", exc_info=True)
        # Return error structure for LLM
        return {"error": f"Error during execution of {function_name}: {e}"}

# --- Deprecated/Placeholder Definitions (If needed for compatibility) ---
# Kept for reference if llm_service still imports them directly
class ToolFunction(BaseModel):
    """Alias for ToolDefinition, if needed."""
    name: str
    description: str
    parameters: List[ToolParameter]