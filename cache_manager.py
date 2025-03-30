"""
Cache manager for Eva backend using Firestore.
Provides caching functionality previously handled by Redis.
"""
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from firestore_manager import store_document, get_document, delete_document

logger = logging.getLogger(__name__)

# Cache collections
CONVERSATION_CACHE = "conversation_cache"
WEATHER_CACHE = "weather_cache"
NEWS_CACHE = "news_cache"
GENERAL_CACHE = "general_cache"

async def cache_conversation(conversation_id: str, history: List[Dict[str, Any]]) -> bool:
    """Cache conversation history in Firestore."""
    try:
        data = {
            "history": history,
            "updated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        return await store_document(CONVERSATION_CACHE, conversation_id, data)
    except Exception as e:
        logger.error(f"Error caching conversation {conversation_id}: {e}")
        return False

async def get_cached_conversation(conversation_id: str) -> Optional[List[Dict[str, Any]]]:
    """Get cached conversation history from Firestore."""
    try:
        data = await get_document(CONVERSATION_CACHE, conversation_id)
        if not data:
            return None
            
        # Check expiration
        if "expires_at" in data:
            expires_at = datetime.fromisoformat(data["expires_at"])
            if datetime.utcnow() > expires_at:
                # Expired cache
                await delete_document(CONVERSATION_CACHE, conversation_id)
                return None
                
        return data.get("history")
    except Exception as e:
        logger.error(f"Error retrieving cached conversation {conversation_id}: {e}")
        return None

async def set_cache(cache_type: str, key: str, data: Any, expires_in_seconds: int = 300) -> bool:
    """Set a cached item in Firestore."""
    try:
        cache_data = {
            "data": data if isinstance(data, (dict, list)) else str(data),
            "updated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires_in_seconds)).isoformat()
        }
        return await store_document(cache_type, key, cache_data)
    except Exception as e:
        logger.error(f"Error setting cache {cache_type}:{key}: {e}")
        return False

async def get_cache(cache_type: str, key: str) -> Optional[Any]:
    """Get a cached item from Firestore."""
    try:
        data = await get_document(cache_type, key)
        if not data:
            return None
            
        # Check expiration
        if "expires_at" in data:
            expires_at = datetime.fromisoformat(data["expires_at"])
            if datetime.utcnow() > expires_at:
                # Expired cache
                await delete_document(cache_type, key)
                return None
                
        return data.get("data")
    except Exception as e:
        logger.error(f"Error retrieving cache {cache_type}:{key}: {e}")
        return None

# Specialized wrappers for weather/news to maintain API compatibility
async def cache_weather(weather_data: Dict[str, Any], expires_in_seconds: int = 300) -> bool:
    """Cache weather data."""
    return await set_cache(WEATHER_CACHE, "current", weather_data, expires_in_seconds)

async def get_cached_weather() -> Optional[Dict[str, Any]]:
    """Get cached weather data."""
    return await get_cache(WEATHER_CACHE, "current")

async def cache_news(news_data: Dict[str, Any], expires_in_seconds: int = 300) -> bool:
    """Cache news data."""
    return await set_cache(NEWS_CACHE, "current", news_data, expires_in_seconds)

async def get_cached_news() -> Optional[Dict[str, Any]]:
    """Get cached news data."""
    return await get_cache(NEWS_CACHE, "current")