from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from settings import settings
import time
from functools import wraps
import redis
from config import REDIS_HOST, REDIS_PORT, REDIS_PASSWORD

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATELIMIT_PER_MINUTE}/minute"]  # Fixed name to match settings
)

redis_client = None

def get_redis_client():
    global redis_client
    if redis_client is None:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            decode_responses=True
        )
    return redis_client

def rate_limit(limit: int, period: int):
    """Rate limiting decorator for API endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get client info from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None:
                for _, arg in kwargs.items():
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if request is None:
                return await func(*args, **kwargs)
            
            # Get client identifier (IP or device ID from headers)
            client_id = request.headers.get("X-Device-Id", request.client.host)
            
            # Apply rate limiting
            redis = get_redis_client()
            key = f"rate_limit:{client_id}:{func.__name__}"
            
            # Get current count
            current = redis.get(key)
            if current is None:
                # First request, set counter to 1 and set expiry
                redis.set(key, 1)
                redis.expire(key, period)
                return await func(*args, **kwargs)
            
            # Increment counter
            current = int(current)
            if current >= limit:
                # Rate limit exceeded
                retry_after = redis.ttl(key)
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Allow request and increment counter
            redis.incr(key)
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

def setup_limiter(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)