# Replace these imports:
import redis.asyncio as redis
from redis_manager import cache_conversation, get_cached_conversation, get_redis_client

# With these:
from cache_manager import cache_conversation, get_cached_conversation
from firestore_manager import (
    store_document, get_document, delete_document, 
    get_if_not_expired, store_with_expiry
)

# Remove this:
redis_client = get_redis_client()

# Replace the verify_device function:
async def verify_device(device_id: str) -> bool:
    """Verify a device is registered and authorized using Firestore"""
    device_data = await get_document("devices", device_id)
    if not device_data:
        logger.warning(f"Unauthorized device: {device_id}")
        return False
    
    if not device_data.get("verified"):
        logger.warning(f"Device not verified: {device_id}")
        return False
    
    return True

# In the chat_endpoint function, replace Redis device check:
# Replace this:
key = f"device:{request_data.device_id}"
device_data_json = await redis_client.get(key)
if not device_data_json:
    logger.warning(f"Unauthorized device: {request_data.device_id}")
    raise AuthenticationError(detail="Device not authorized")
device_data = json.loads(device_data_json)
if not device_data.get("verified"):
    logger.warning(f"Device not verified: {request_data.device_id}")
    raise AuthenticationError(detail="Device not authorized")

# With this:
device_data = await get_document("devices", request_data.device_id)
if not device_data:
    logger.warning(f"Unauthorized device: {request_data.device_id}")
    raise AuthenticationError(detail="Device not authorized")
if not device_data.get("verified"):
    logger.warning(f"Device not verified: {request_data.device_id}")
    raise AuthenticationError(detail="Device not authorized")