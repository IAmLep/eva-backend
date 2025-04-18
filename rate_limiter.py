"""
Rate Limiting module for the EVA backend application.

Provides dependencies or middleware for enforcing API rate limits.
This version uses a simple in-memory store.
"""

import time
import logging
from typing import Dict, Optional, Tuple, Annotated
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status

# --- Local Imports ---
from config import get_settings
from exceptions import RateLimitError
from models import User # Needed if limiting per user
from auth import get_current_active_user # Dependency to identify user

logger = logging.getLogger(__name__)

# --- In-Memory Rate Limiting Store ---
# Structure: {identifier: (last_request_timestamp, minute_count, day_count)}
# Identifier could be user_id, api_key_id, or IP address
rate_limit_store: Dict[str, Tuple[float, int, int]] = defaultdict(lambda: (0.0, 0, 0))

# --- Rate Limiter Class (Dependency) ---
class RateLimiter:
    """
    FastAPI dependency class for enforcing rate limits based on settings.
    Limits are applied per user.
    """
    def __init__(self):
        settings = get_settings()
        # Load limits from settings
        self.requests_per_minute = settings.RATE_LIMIT_PER_MINUTE
        self.requests_per_day = settings.RATE_LIMIT_PER_DAY
        logger.info(f"Rate limiter initialized: {self.requests_per_minute}/min, {self.requests_per_day}/day")

    async def __call__(
        self,
        request: Request, # Inject request for IP-based limiting if needed
        user: Annotated[User, Depends(get_current_active_user)] # Require authenticated user
    ):
        """Checks and updates rate limits for the identified user."""
        # Use user ID as the identifier for rate limiting
        identifier = user.id
        current_time = time.time()

        # Get current counts for the identifier
        last_request_time, minute_count, day_count = rate_limit_store[identifier]

        # --- Calculate Time Windows ---
        minute_window_start = current_time - 60
        day_window_start = current_time - 86400 # 24 * 60 * 60

        # --- Reset Counts if Windows Expired ---
        if last_request_time < minute_window_start:
            minute_count = 0 # Reset minute count
        if last_request_time < day_window_start:
            day_count = 0 # Reset day count

        # --- Check Limits ---
        if minute_count >= self.requests_per_minute:
            retry_after = int(last_request_time + 60 - current_time) + 1
            logger.warning(f"Rate limit exceeded (minute) for user {identifier}. Count: {minute_count+1}")
            raise RateLimitError(
                detail=f"Minute rate limit exceeded. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )

        if day_count >= self.requests_per_day:
            retry_after = int(last_request_time + 86400 - current_time) + 1
            logger.warning(f"Rate limit exceeded (day) for user {identifier}. Count: {day_count+1}")
            raise RateLimitError(
                detail=f"Daily rate limit exceeded. Try again later.",
                # Retry-After for daily limits might be very long, consider omitting or setting reasonable max
                # headers={"Retry-After": str(retry_after)}
            )

        # --- Update Counts ---
        new_minute_count = minute_count + 1
        new_day_count = day_count + 1
        rate_limit_store[identifier] = (current_time, new_minute_count, new_day_count)

        logger.debug(f"Rate limit check passed for user {identifier}. Counts: {new_minute_count}/min, {new_day_count}/day")

        # Optionally, add rate limit headers to the response (requires middleware or accessing response object)
        # This dependency cannot easily modify the response headers.
        # Consider adding headers in a middleware if needed.

# --- Dependency Instance ---
# Create an instance of the RateLimiter class to use as a dependency
rate_limiter_dependency = RateLimiter()

# --- How to use in routers ---
# from fastapi import APIRouter, Depends
# from .rate_limiter import rate_limiter_dependency
#
# router = APIRouter(dependencies=[Depends(rate_limiter_dependency)])
#
# @router.get("/limited-resource")
# async def get_limited_resource():
#     # If code reaches here, rate limit check passed
#     return {"message": "You have access!"}
