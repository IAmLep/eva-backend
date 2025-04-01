"""
API Tools module for EVA backend.

This module provides tools and utilities for function calling
integration with Gemini API, allowing LLM to execute actions.

Last updated: 2025-04-01
Version: v1.7 (redis removal)
"""

import inspect
import json
import logging
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, Field, create_model

from auth import get_current_user
from config import get_settings
from database import get_db_manager
from exceptions import FunctionCallError
from models import User

# Logger configuration
logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """
    Tool parameter definition for function calling.
    
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


class ToolDefinition(BaseModel):
    """
    Tool definition for function calling.
    
    Attributes:
        name: Function name
        description: Function description
        parameters: List of parameters
    """
    name: str
    description: str
    parameters: List[ToolParameter]


class ToolCall(BaseModel):
    """
    Tool call from LLM.
    
    Attributes:
        function: Tool definition
        args: Arguments for the function
    """
    function: ToolDefinition
    args: Dict[str, Any]


class Tool:
    """Base class for all tools that can be called by the LLM."""
    
    @staticmethod
    def get_definition() -> ToolDefinition:
        """
        Get tool definition.
        
        Returns:
            ToolDefinition: Tool definition for LLM
        """
        raise NotImplementedError("Tools must implement get_definition")
    
    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            args: Tool arguments
            user: Current user
            
        Returns:
            Dict[str, Any]: Execution result
            
        Raises:
            FunctionCallError: If execution fails
        """
        raise NotImplementedError("Tools must implement execute")


class TimeTool(Tool):
    """Tool for getting current time and date information."""
    
    @staticmethod
    def get_definition() -> ToolDefinition:
        """
        Get tool definition.
        
        Returns:
            ToolDefinition: Tool definition for LLM
        """
        return ToolDefinition(
            name="get_time",
            description="Get the current time and date information",
            parameters=[
                ToolParameter(
                    name="format",
                    type="string",
                    description="Format of the time/date (iso, human, unix)",
                    required=False,
                    enum=["iso", "human", "unix"]
                ),
                ToolParameter(
                    name="timezone",
                    type="string",
                    description="Timezone for the time (utc, local)",
                    required=False,
                    enum=["utc", "local"]
                )
            ]
        )
    
    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            args: Tool arguments
            user: Current user
            
        Returns:
            Dict[str, Any]: Current time information
        """
        from datetime import datetime
        import time
        
        time_format = args.get("format", "iso")
        timezone = args.get("timezone", "utc")
        
        try:
            if timezone == "utc":
                now = datetime.utcnow()
            else:
                now = datetime.now()
            
            if time_format == "iso":
                time_str = now.isoformat()
            elif time_format == "human":
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
            elif time_format == "unix":
                time_str = str(int(time.time()))
            else:
                time_str = now.isoformat()
            
            return {
                "success": True,
                "time": time_str,
                "format": time_format,
                "timezone": timezone,
                "append_to_response": False
            }
        except Exception as e:
            logger.error(f"Error executing time tool: {str(e)}")
            raise FunctionCallError(f"Failed to get time information: {str(e)}")


class WeatherTool(Tool):
    """Tool for getting weather information for a location."""
    
    @staticmethod
    def get_definition() -> ToolDefinition:
        """
        Get tool definition.
        
        Returns:
            ToolDefinition: Tool definition for LLM
        """
        return ToolDefinition(
            name="get_weather",
            description="Get the current weather information for a location",
            parameters=[
                ToolParameter(
                    name="location",
                    type="string",
                    description="Location to get weather for (city name or coordinates)",
                    required=True
                ),
                ToolParameter(
                    name="units",
                    type="string",
                    description="Units for temperature (celsius, fahrenheit)",
                    required=False,
                    enum=["celsius", "fahrenheit"]
                )
            ]
        )
    
    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            args: Tool arguments
            user: Current user
            
        Returns:
            Dict[str, Any]: Weather information
            
        Raises:
            FunctionCallError: If execution fails
        """
        try:
            # This is a mock implementation
            # In a real implementation, you would call a weather API
            location = args.get("location", "")
            units = args.get("units", "celsius")
            
            if not location:
                raise FunctionCallError("Location is required")
            
            # Mock weather data
            weather_data = {
                "location": location,
                "temperature": 22 if units == "celsius" else 72,
                "units": units,
                "condition": "Sunny",
                "humidity": 65,
                "wind_speed": 10,
                "precipitation": 0
            }
            
            return {
                "success": True,
                "weather": weather_data,
                "append_to_response": True,
                "message": f"The current weather in {location} is {weather_data['condition']} with a temperature of {weather_data['temperature']}°{'C' if units == 'celsius' else 'F'}."
            }
        except FunctionCallError:
            # Re-raise function call errors
            raise
        except Exception as e:
            logger.error(f"Error executing weather tool: {str(e)}")
            raise FunctionCallError(f"Failed to get weather information: {str(e)}")


class MemoryTool(Tool):
    """Tool for managing user memories."""
    
    @staticmethod
    def get_definition() -> ToolDefinition:
        """
        Get tool definition.
        
        Returns:
            ToolDefinition: Tool definition for LLM
        """
        return ToolDefinition(
            name="manage_memory",
            description="Save or retrieve memories for the user",
            parameters=[
                ToolParameter(
                    name="action",
                    type="string",
                    description="Action to perform (save, retrieve, delete)",
                    required=True,
                    enum=["save", "retrieve", "delete"]
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Memory content to save (required for 'save' action)",
                    required=False
                ),
                ToolParameter(
                    name="query",
                    type="string",
                    description="Query to search for memories (for 'retrieve' action)",
                    required=False
                ),
                ToolParameter(
                    name="memory_id",
                    type="string",
                    description="Memory ID to delete (required for 'delete' action)",
                    required=False
                )
            ]
        )
    
    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            args: Tool arguments
            user: Current user
            
        Returns:
            Dict[str, Any]: Execution result
            
        Raises:
            FunctionCallError: If execution fails
        """
        try:
            action = args.get("action", "")
            
            if not action:
                raise FunctionCallError("Action is required")
            
            db = get_db_manager()
            
            if action == "save":
                content = args.get("content", "")
                if not content:
                    raise FunctionCallError("Content is required for 'save' action")
                
                # Create memory
                from models import Memory
                memory = Memory(
                    user_id=user.id,
                    content=content
                )
                
                memory_id = await db.create_memory(memory)
                
                return {
                    "success": True,
                    "memory_id": memory_id,
                    "action": "save",
                    "append_to_response": True,
                    "message": "Memory saved successfully."
                }
            
            elif action == "retrieve":
                query = args.get("query", "")
                if not query:
                    raise FunctionCallError("Query is required for 'retrieve' action")
                
                # Search memories
                memories = await db.get_memories_by_query(user.id, query)
                
                if not memories:
                    return {
                        "success": True,
                        "memories": [],
                        "action": "retrieve",
                        "append_to_response": True,
                        "message": "No memories found matching your query."
                    }
                
                # Format memories for response
                memory_list = [
                    {"id": mem.memory_id, "content": mem.content, "created_at": mem.created_at.isoformat()}
                    for mem in memories[:5]  # Limit to 5 memories
                ]
                
                return {
                    "success": True,
                    "memories": memory_list,
                    "count": len(memories),
                    "action": "retrieve",
                    "append_to_response": True,
                    "message": f"Found {len(memories)} memories. Here are the most relevant ones:\n" + 
                              "\n".join([f"- {m['content']}" for m in memory_list])
                }
            
            elif action == "delete":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    raise FunctionCallError("Memory ID is required for 'delete' action")
                
                # Delete memory
                success = await db.delete_memory(user.id, memory_id)
                
                if success:
                    return {
                        "success": True,
                        "memory_id": memory_id,
                        "action": "delete",
                        "append_to_response": True,
                        "message": "Memory deleted successfully."
                    }
                else:
                    return {
                        "success": False,
                        "memory_id": memory_id,
                        "action": "delete",
                        "append_to_response": True,
                        "message": "Failed to delete memory. It may not exist or you don't have permission."
                    }
            
            else:
                raise FunctionCallError(f"Unknown action: {action}")
        
        except FunctionCallError:
            # Re-raise function call errors
            raise
        except Exception as e:
            logger.error(f"Error executing memory tool: {str(e)}")
            raise FunctionCallError(f"Failed to manage memory: {str(e)}")


class SyncTool(Tool):
    """Tool for triggering database synchronization."""
    
    @staticmethod
    def get_definition() -> ToolDefinition:
        """
        Get tool definition.
        
        Returns:
            ToolDefinition: Tool definition for LLM
        """
        return ToolDefinition(
            name="sync_database",
            description="Trigger database synchronization with cloud",
            parameters=[
                ToolParameter(
                    name="full_sync",
                    type="boolean",
                    description="Whether to perform a full sync (true) or incremental sync (false)",
                    required=False
                ),
                ToolParameter(
                    name="cleanup",
                    type="boolean",
                    description="Whether to clean up old entries after sync",
                    required=False
                )
            ]
        )
    
    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            args: Tool arguments
            user: Current user
            
        Returns:
            Dict[str, Any]: Execution result
            
        Raises:
            FunctionCallError: If execution fails
        """
        try:
            full_sync = args.get("full_sync", False)
            cleanup = args.get("cleanup", False)
            
            # Import here to avoid circular imports
            from api_sync import trigger_sync
            
            # Call the sync function
            result = await trigger_sync(user.id, full_sync=full_sync, cleanup=cleanup)
            
            if result.get("success", False):
                return {
                    "success": True,
                    "synced_items": result.get("synced_items", 0),
                    "deleted_items": result.get("deleted_items", 0),
                    "full_sync": full_sync,
                    "cleanup": cleanup,
                    "append_to_response": True,
                    "message": f"Database sync completed successfully. Synced {result.get('synced_items', 0)} items." +
                              (f" Cleaned up {result.get('deleted_items', 0)} old items." if cleanup else "")
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error"),
                    "append_to_response": True,
                    "message": f"Database sync failed: {result.get('error', 'Unknown error')}"
                }
        
        except Exception as e:
            logger.error(f"Error executing sync tool: {str(e)}")
            raise FunctionCallError(f"Failed to sync database: {str(e)}")


class CleanupTool(Tool):
    """Tool for cleaning up old or duplicate memories."""
    
    @staticmethod
    def get_definition() -> ToolDefinition:
        """
        Get tool definition.
        
        Returns:
            ToolDefinition: Tool definition for LLM
        """
        return ToolDefinition(
            name="cleanup_database",
            description="Clean up old or duplicate entries in the database",
            parameters=[
                ToolParameter(
                    name="cleanup_type",
                    type="string",
                    description="Type of cleanup to perform",
                    required=True,
                    enum=["old", "duplicates", "both"]
                ),
                ToolParameter(
                    name="days_threshold",
                    type="integer",
                    description="Age threshold in days for old entries (for 'old' or 'both' types)",
                    required=False
                )
            ]
        )
    
    @staticmethod
    async def execute(args: Dict[str, Any], user: User) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Args:
            args: Tool arguments
            user: Current user
            
        Returns:
            Dict[str, Any]: Execution result
            
        Raises:
            FunctionCallError: If execution fails
        """
        try:
            cleanup_type = args.get("cleanup_type", "")
            days_threshold = args.get("days_threshold", 30)
            
            if not cleanup_type:
                raise FunctionCallError("Cleanup type is required")
            
            db = get_db_manager()
            old_count = 0
            duplicates_count = 0
            
            if cleanup_type in ["old", "both"]:
                # Clean up old memories
                old_count = await db.cleanup_old_memories(user.id, days_threshold)
            
            if cleanup_type in ["duplicates", "both"]:
                # Clean up duplicate memories
                from firestore_manager import get_firestore_client
                firestore_manager = get_firestore_client().parent
                duplicates_count = await firestore_manager.cleanup_duplicate_memories(user.id)
            
            total_count = old_count + duplicates_count
            
            return {
                "success": True,
                "cleanup_type": cleanup_type,
                "old_count": old_count,
                "duplicates_count": duplicates_count,
                "total_count": total_count,
                "days_threshold": days_threshold,
                "append_to_response": True,
                "message": f"Database cleanup completed. " +
                          (f"Removed {old_count} old entries. " if old_count > 0 else "") +
                          (f"Removed {duplicates_count} duplicate entries." if duplicates_count > 0 else "") +
                          (f" No entries were removed." if total_count == 0 else "")
            }
        
        except FunctionCallError:
            # Re-raise function call errors
            raise
        except Exception as e:
            logger.error(f"Error executing cleanup tool: {str(e)}")
            raise FunctionCallError(f"Failed to clean up database: {str(e)}")


# Register available tools
_available_tools = [
    TimeTool,
    WeatherTool,
    MemoryTool,
    SyncTool,
    CleanupTool
]


def available_tools() -> List[ToolDefinition]:
    """
    Get list of available tools.
    
    Returns:
        List[ToolDefinition]: List of tool definitions
    """
    return [tool.get_definition() for tool in _available_tools]


def get_tool_by_name(name: str) -> Optional[Type[Tool]]:
    """
    Get tool class by name.
    
    Args:
        name: Tool name
        
    Returns:
        Optional[Type[Tool]]: Tool class if found
    """
    for tool in _available_tools:
        if tool.get_definition().name == name:
            return tool
    return None


async def execute_function_call(call: ToolCall, user: User) -> Dict[str, Any]:
    """
    Execute a function call from LLM.
    
    Args:
        call: Tool call information
        user: Current user
        
    Returns:
        Dict[str, Any]: Execution result
        
    Raises:
        FunctionCallError: If execution fails
    """
    try:
        # Get function name
        function_name = call.function.name
        
        # Get tool
        tool = get_tool_by_name(function_name)
        if not tool:
            logger.error(f"Unknown function: {function_name}")
            raise FunctionCallError(f"Unknown function: {function_name}")
        
        # Execute tool
        result = await tool.execute(call.args, user)
        
        logger.info(f"Executed function {function_name} for user {user.username}")
        return result
    
    except FunctionCallError:
        # Re-raise function call errors
        raise
    except Exception as e:
        logger.error(f"Error executing function call: {str(e)}")
        raise FunctionCallError(f"Function execution error: {str(e)}")


def register_tool(tool_class: Type[Tool]):
    """
    Register a new tool.
    
    Args:
        tool_class: Tool class to register
    """
    global _available_tools
    
    # Check if tool already registered
    for existing_tool in _available_tools:
        if existing_tool.get_definition().name == tool_class.get_definition().name:
            logger.warning(f"Tool {tool_class.get_definition().name} already registered")
            return
    
    # Add tool
    _available_tools.append(tool_class)
    logger.info(f"Registered tool: {tool_class.get_definition().name}")