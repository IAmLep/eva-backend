"""
Rate limiting for the Eva backend API.
Uses Firestore for distributed rate limiting with fallback to in-memory.
"""
from fastapi import HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from settings import settings
from datetime import datetime, timedelta
import time
from functools import wraps
import logging
import asyncio

from firestore_manager import get_rate_limit, update_rate_limit

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize the limiter at module level and make it available for import
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.RATELIMIT_PER_MINUTE}/minute"]
)

# Add __all__ to explicitly export these names
__all__ = ['limiter', 'rate_limit', 'setup_limiter']

# In-memory rate limit store as fallback
_rate_limit_store = {}

async def _check_rate_limit(key: str, limit: int, period: int) -> bool:
    """
    Check if a request should be rate limited.
    Returns True if request is allowed, False if rate limited.
    """
    try:
        # Try Firestore first
        count, expires_at = await get_rate_limit(key)
        current_time = datetime.utcnow()
        
        # Check if expired
        if current_time > expires_at:
            # Reset counter
            new_expires = current_time + timedelta(seconds=period)
            await update_rate_limit(key, 1, new_expires)
            return True
        elif count >= limit:
            # Rate limit exceeded
            retry_after = int((expires_at - current_time).total_seconds())
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
        else:
            # Increment counter
            await update_rate_limit(key, count + 1, expires_at)
            return True
            
    except HTTPException:
        # Re-raise HTTP exceptions (rate limit)
        raise
    except Exception as e:
        logger.error(f"Firestore rate limiting error: {e}, falling back to in-memory")
        # Fall back to in-memory rate limiting
        return _check_memory_rate_limit(key, limit, period)

def _check_memory_rate_limit(key: str, limit: int, period: int) -> bool:
    """In-memory rate limiting fallback, returns True if allowed."""
    current_time = time.time()
    
    if key in _rate_limit_store:
        count, expires = _rate_limit_store[key]
        
        if current_time > expires:
            # Reset counter
            _rate_limit_store[key] = (1, current_time + period)
            return True
        elif count >= limit:
            # Rate limit exceeded
            retry_after = int(expires - current_time)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
        else:
            # Increment counter
            _rate_limit_store[key] = (count + 1, expires)
            return True
    else:
        # First request
        _rate_limit_store[key] = (1, current_time + period)
        return True

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
                # No request found, skip rate limiting
                return await func(*args, **kwargs)
            
            # Get client identifier (IP or device ID from headers)
            client_id = request.headers.get("X-Device-Id") or request.client.host
            
            # Apply rate limiting
            key = f"rate_limit:{client_id}:{func.__name__}"
            
            # Check rate limit
            await _check_rate_limit(key, limit, period)
            
            # If we get here, request is allowed
            return await func(*args, **kwargs)
            
        return wrapper
    return decorator

def setup_limiter(app):
    """Configure the rate limiter for the FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)