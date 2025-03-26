import asyncio
import logging
import json
from typing import Optional
import redis.asyncio as redis
from settings import settings

logger = logging.getLogger(__name__)

class RedisManager:
    _instance: Optional[redis.Redis] = None
    
    @classmethod
    async def get_instance(cls) -> redis.Redis:
        if cls._instance is None:
            config = {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB,
                "decode_responses": True
            }
            
            # Only add password if it's not empty
            if settings.REDIS_PASSWORD:
                config["password"] = settings.REDIS_PASSWORD
                
            try:
                cls._instance = redis.Redis(**config)
                # Test the connection
                await cls._instance.ping()
                logger.info("Successfully connected to Redis")
            except redis.ConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
            
        return cls._instance

redis_client = None

def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD,
            decode_responses=True
        )
    return redis_client

def cache_conversation(conversation_id, messages, expiry=3600):
    """Cache conversation history in Redis"""
    client = get_redis_client()
    key = f"conversation:{conversation_id}"
    client.set(key, json.dumps(messages))
    client.expire(key, expiry)

def get_cached_conversation(conversation_id):
    """Retrieve cached conversation history from Redis"""
    client = get_redis_client()
    key = f"conversation:{conversation_id}"
    data = client.get(key)
    if data:
        return json.loads(data)
    return None

def clear_conversation_cache(conversation_id):
    """Clear a conversation from cache"""
    client = get_redis_client()
    key = f"conversation:{conversation_id}"
    client.delete(key)