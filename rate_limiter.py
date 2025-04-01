"""
Rate Limiter module for EVA backend.

This module provides rate limiting functionality to protect API endpoints
from excessive usage and ensure fair resource allocation.

Last updated: 2025-04-01 10:47:17
Version: v1.8.6
Created by: IAmLep
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

from fastapi import Depends, HTTPException, Request, Response, status
from pydantic import BaseModel

from config import get_settings
from exceptions import RateLimitError
from models import RateLimitConfig, RateLimitTier, UserRateLimit

# Logger configuration
logger = logging.getLogger(__name__)


class RateLimitType(str, Enum):
    """
    Rate limit type enumeration.
    
    Defines the different types of rate limits.
    
    Attributes:
        REQUESTS: Limit based on request count
        TOKENS: Limit based on token count
    """
    REQUESTS = "requests"
    TOKENS = "tokens"


class RateLimitWindow(str, Enum):
    """
    Rate limit window enumeration.
    
    Defines the time windows for rate limiting.
    
    Attributes:
        MINUTE: Per minute rate limiting
        DAY: Per day rate limiting
    """
    MINUTE = "minute"
    DAY = "day"


class RateLimitResult(BaseModel):
    """
    Rate limit check result model.
    
    Contains the result of a rate limit check.
    
    Attributes:
        allowed: Whether the request is allowed
        current: Current usage count
        limit: Maximum allowed count
        remaining: Remaining available count
        reset_at: When the current window resets
        window: Rate limit window
        limit_type: Type of rate limit
    """
    allowed: bool
    current: int
    limit: int
    remaining: int
    reset_at: datetime
    window: RateLimitWindow
    limit_type: RateLimitType


class RateLimiter:
    """
    Rate limiter for API requests.
    
    This class provides in-memory rate limiting with configurable
    tiers, windows, and limits.
    """
    
    def __init__(self):
        """Initialize the rate limiter with default configurations."""
        self.settings = get_settings()
        
        # Define rate limit configurations for different tiers
        self.limit_configs: Dict[RateLimitTier, RateLimitConfig] = {
            RateLimitTier.FREE: RateLimitConfig(
                tier=RateLimitTier.FREE,
                requests_per_minute=self.settings.RATE_LIMIT_PER_MINUTE,
                requests_per_day=self.settings.RATE_LIMIT_PER_DAY,
                tokens_per_minute=10000,
                tokens_per_day=50000
            ),
            RateLimitTier.STANDARD: RateLimitConfig(
                tier=RateLimitTier.STANDARD,
                requests_per_minute=self.settings.RATE_LIMIT_PER_MINUTE * 2,
                requests_per_day=self.settings.RATE_LIMIT_PER_DAY * 2,
                tokens_per_minute=20000,
                tokens_per_day=100000
            ),
            RateLimitTier.PREMIUM: RateLimitConfig(
                tier=RateLimitTier.PREMIUM,
                requests_per_minute=self.settings.RATE_LIMIT_PER_MINUTE * 5,
                requests_per_day=self.settings.RATE_LIMIT_PER_DAY * 5,
                tokens_per_minute=50000,
                tokens_per_day=250000
            )
        }
        
        # User rate limit tracking
        self.user_limits: Dict[str, UserRateLimit] = {}
        
        logger.info("Rate limiter initialized")
    
    def get_user_limit(self, user_id: str, tier: RateLimitTier = RateLimitTier.FREE) -> UserRateLimit:
        """
        Get or create rate limit tracker for a user.
        
        Args:
            user_id: User ID
            tier: Rate limit tier
            
        Returns:
            UserRateLimit: User rate limit tracker
        """
        if user_id not in self.user_limits:
            self.user_limits[user_id] = UserRateLimit(
                user_id=user_id,
                tier=tier
            )
        
        # Update tier if it has changed
        if self.user_limits[user_id].tier != tier:
            self.user_limits[user_id].tier = tier
        
        return self.user_limits[user_id]
    
    def check_rate_limit(
        self, 
        user_id: str, 
        limit_type: RateLimitType = RateLimitType.REQUESTS,
        window: RateLimitWindow = RateLimitWindow.MINUTE,
        increment: int = 1,
        tier: Optional[RateLimitTier] = None
    ) -> RateLimitResult:
        """
        Check if a request is within rate limits.
        
        Args:
            user_id: User ID
            limit_type: Type of rate limit to check
            window: Time window for rate limiting
            increment: Amount to increment the counter
            tier: Optional rate limit tier override
            
        Returns:
            RateLimitResult: Rate limit check result
        """
        # Get user limits
        user_limit = self.get_user_limit(user_id, tier or RateLimitTier.FREE)
        
        # Get config for user's tier
        config = self.limit_configs[user_limit.tier]
        
        # Reset counters if needed
        self._reset_counters_if_needed(user_limit)
        
        # Determine limit, current count, and counter to increment
        if limit_type == RateLimitType.REQUESTS:
            if window == RateLimitWindow.MINUTE:
                limit = config.requests_per_minute
                current = user_limit.minute_count
                counter_attr = "minute_count"
                reset_at = user_limit.last_reset_minute + timedelta(minutes=1)
            else:  # DAY
                limit = config.requests_per_day
                current = user_limit.day_count
                counter_attr = "day_count"
                reset_at = user_limit.last_reset_day + timedelta(days=1)
        else:  # TOKENS
            if window == RateLimitWindow.MINUTE:
                limit = config.tokens_per_minute
                current = user_limit.minute_tokens
                counter_attr = "minute_tokens"
                reset_at = user_limit.last_reset_minute + timedelta(minutes=1)
            else:  # DAY
                limit = config.tokens_per_day
                current = user_limit.day_tokens
                counter_attr = "day_tokens"
                reset_at = user_limit.last_reset_day + timedelta(days=1)
        
        # Check if limit would be exceeded
        new_count = current + increment
        is_allowed = new_count <= limit
        
        # Increment counter if allowed
        if is_allowed:
            setattr(user_limit, counter_attr, new_count)
        
        # Calculate remaining
        remaining = max(0, limit - new_count)
        
        # Create result
        result = RateLimitResult(
            allowed=is_allowed,
            current=new_count if is_allowed else current,
            limit=limit,
            remaining=remaining,
            reset_at=reset_at,
            window=window,
            limit_type=limit_type
        )
        
        # Log if rate limited
        if not is_allowed:
            logger.warning(
                f"Rate limit exceeded for user {user_id}: "
                f"{limit_type.value} limit of {limit} per {window.value} "
                f"(current: {current}, requested: {increment})"
            )
        
        return result
    
    def _reset_counters_if_needed(self, user_limit: UserRateLimit) -> None:
        """
        Reset rate limit counters if their time windows have passed.
        
        Args:
            user_limit: User rate limit tracker to check
        """
        now = datetime.utcnow()
        
        # Check minute window
        minute_ago = now - timedelta(minutes=1)
        if user_limit.last_reset_minute < minute_ago:
            user_limit.minute_count = 0
            user_limit.minute_tokens = 0
            user_limit.last_reset_minute = now
        
        # Check day window
        day_ago = now - timedelta(days=1)
        if user_limit.last_reset_day < day_ago:
            user_limit.day_count = 0
            user_limit.day_tokens = 0
            user_limit.last_reset_day = now


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get rate limiter singleton instance.
    
    Returns:
        RateLimiter: Rate limiter instance
    """
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


def rate_limit(
    limit_type: RateLimitType = RateLimitType.REQUESTS,
    window: RateLimitWindow = RateLimitWindow.MINUTE,
    increment: int = 1
):
    """
    Decorator for rate limiting API endpoints.
    
    Args:
        limit_type: Type of rate limit to apply
        window: Time window for rate limiting
        increment: Amount to increment the counter
        
    Returns:
        Callable: Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get rate limiter
            rate_limiter = get_rate_limiter()
            
            # Extract request and user ID
            request = None
            user_id = None
            
            # Find request in args or kwargs
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None and "request" in kwargs:
                request = kwargs["request"]
            
            # Get user ID from request state
            if request is not None and hasattr(request.state, "user_id"):
                user_id = request.state.user_id
            
            # Fallback to a default user ID if not found
            if user_id is None:
                client_ip = request.client.host if request and request.client else "unknown"
                user_id = f"anonymous_{client_ip}"
            
            # Check rate limit
            result = rate_limiter.check_rate_limit(
                user_id=user_id,
                limit_type=limit_type,
                window=window,
                increment=increment
            )
            
            # Add rate limit headers if there's a response in kwargs
            if "response" in kwargs and hasattr(kwargs["response"], "headers"):
                response = kwargs["response"]
                response.headers["X-RateLimit-Limit"] = str(result.limit)
                response.headers["X-RateLimit-Remaining"] = str(result.remaining)
                response.headers["X-RateLimit-Reset"] = str(int(result.reset_at.timestamp()))
            
            # Raise error if rate limited
            if not result.allowed:
                reset_in_seconds = int((result.reset_at - datetime.utcnow()).total_seconds())
                raise RateLimitError(
                    detail=f"Rate limit exceeded. Try again in {reset_in_seconds} seconds.",
                    reset_at=str(result.reset_at.isoformat())
                )
            
            # Call the original function
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Similar logic for synchronous functions
            rate_limiter = get_rate_limiter()
            
            request = None
            user_id = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request is None and "request" in kwargs:
                request = kwargs["request"]
            
            if request is not None and hasattr(request.state, "user_id"):
                user_id = request.state.user_id
            
            if user_id is None:
                client_ip = request.client.host if request and request.client else "unknown"
                user_id = f"anonymous_{client_ip}"
            
            result = rate_limiter.check_rate_limit(
                user_id=user_id,
                limit_type=limit_type,
                window=window,
                increment=increment
            )
            
            if "response" in kwargs and hasattr(kwargs["response"], "headers"):
                response = kwargs["response"]
                response.headers["X-RateLimit-Limit"] = str(result.limit)
                response.headers["X-RateLimit-Remaining"] = str(result.remaining)
                response.headers["X-RateLimit-Reset"] = str(int(result.reset_at.timestamp()))
            
            if not result.allowed:
                reset_in_seconds = int((result.reset_at - datetime.utcnow()).total_seconds())
                raise RateLimitError(
                    detail=f"Rate limit exceeded. Try again in {reset_in_seconds} seconds.",
                    reset_at=str(result.reset_at.isoformat())
                )
            
            return func(*args, **kwargs)
        
        # Determine if function is async or sync
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def check_token_limit(text: str, limit: int = 1000) -> bool:
    """
    Check if text is within token limit (approximate).
    
    Uses a simple approximation of 4 characters per token.
    
    Args:
        text: Text to check
        limit: Token limit
        
    Returns:
        bool: True if within limit, False otherwise
    """
    # Simple approximation: ~4 chars per token
    estimated_tokens = len(text) / 4
    return estimated_tokens <= limit


def require_rate_limit_check(
    limit_type: RateLimitType = RateLimitType.REQUESTS,
    window: RateLimitWindow = RateLimitWindow.MINUTE,
    increment: int = 1
):
    """
    Dependency for rate limiting endpoints.
    
    Args:
        limit_type: Type of rate limit to apply
        window: Time window for rate limiting
        increment: Amount to increment the counter
        
    Returns:
        Callable: FastAPI dependency
    """
    async def dependency(request: Request) -> None:
        """
        Check rate limit for the request.
        
        Args:
            request: FastAPI request
            
        Raises:
            RateLimitError: If rate limited
        """
        rate_limiter = get_rate_limiter()
        
        # Get user ID from request state
        user_id = None
        if hasattr(request.state, "user_id"):
            user_id = request.state.user_id
        
        # Fallback to client IP
        if user_id is None:
            client_ip = request.client.host if request.client else "unknown"
            user_id = f"anonymous_{client_ip}"
        
        # Check rate limit
        result = rate_limiter.check_rate_limit(
            user_id=user_id,
            limit_type=limit_type,
            window=window,
            increment=increment
        )
        
        # Add rate limit headers
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(int(result.reset_at.timestamp()))
        }
        
        # Store result in request state
        request.state.rate_limit_result = result
        
        # Raise error if rate limited
        if not result.allowed:
            reset_in_seconds = int((result.reset_at - datetime.utcnow()).total_seconds())
            raise RateLimitError(
                detail=f"Rate limit exceeded. Try again in {reset_in_seconds} seconds.",
                reset_at=str(result.reset_at.isoformat())
            )
    
    return dependency


def apply_rate_limit_headers(request: Request, response: Response) -> None:
    """
    Apply rate limit headers to a response.
    
    Args:
        request: FastAPI request
        response: FastAPI response
    """
    if hasattr(request.state, "rate_limit_headers"):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value