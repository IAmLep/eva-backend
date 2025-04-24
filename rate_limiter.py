"""
Rate Limiting module for the EVA backend application.

Provides dependencies or middleware for enforcing API rate limits.
This version uses a simple in-memory store and limits per user based on config.
"""

import time
import logging
from typing import Dict, Optional, Tuple, Annotated
from collections import defaultdict

from fastapi import Depends, HTTPException, Request, status

# --- Local Imports ---
from config import settings
from exceptions import RateLimitError # Assuming you have this custom exception defined
from models import User # Needed for user identification
from auth import get_current_active_user # Dependency to identify user

logger = logging.getLogger(__name__)

# --- In-Memory Rate Limiting Store ---
# Structure: {identifier: (last_request_timestamp, count_in_window)}
# Identifier is the user_id
rate_limit_store: Dict[str, Tuple[float, int]] = defaultdict(lambda: (0.0, 0))

# --- Rate Limiter Class (Dependency) ---
class RateLimiter:
    """
    FastAPI dependency class for enforcing rate limits based on settings.
    Limits are applied per user based on RATE_LIMIT_USER_REQUESTS and
    RATE_LIMIT_USER_WINDOW_SECONDS.
    """
    def __init__(self):
        
        # Load limits from existing settings in config.py
        self.max_requests = settings.RATE_LIMIT_USER_REQUESTS
        self.window_seconds = settings.RATE_LIMIT_USER_WINDOW_SECONDS
        if not settings.RATE_LIMIT_ENABLED:
             logger.warning("Rate limiting is globally disabled in settings.")
        logger.info(f"Rate limiter initialized: {self.max_requests} requests / {self.window_seconds} seconds per user.")

    async def __call__(
        self,
        request: Request, # Inject request (not used here, but available)
        user: Annotated[User, Depends(get_current_active_user)] # Require authenticated user
    ):
        """Checks and updates rate limits for the identified user."""
         # Get settings again in case needed
        if not settings.RATE_LIMIT_ENABLED:
            logger.debug(f"Rate limit check skipped for user {user.id} (globally disabled).")
            return # Skip check if rate limiting is disabled

        # Use user ID as the identifier for rate limiting
        identifier = user.id
        current_time = time.time()

        # Get current counts for the identifier
        last_request_time, count_in_window = rate_limit_store[identifier]

        # --- Calculate Time Window Start ---
        window_start_time = current_time - self.window_seconds

        # --- Reset Count if Window Expired ---
        if last_request_time < window_start_time:
            count_in_window = 0 # Reset count

        # --- Check Limit ---
        if count_in_window >= self.max_requests:
            # Calculate when the window resets for the Retry-After header
            retry_after = int(last_request_time + self.window_seconds - current_time) + 1
            logger.warning(
                f"Rate limit exceeded for user {identifier}. "
                f"Count: {count_in_window + 1}/{self.max_requests} in {self.window_seconds}s window."
            )
            # Use your custom RateLimitError or FastAPI's HTTPException
            # Using HTTPException for simplicity if RateLimitError isn't defined
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)}
            )
            # If using RateLimitError:
            # raise RateLimitError(
            #     detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            #     headers={"Retry-After": str(retry_after)}
            # )

        # --- Update Count ---
        new_count = count_in_window + 1
        # Store the current time as the last request time for this identifier
        rate_limit_store[identifier] = (current_time, new_count)

        logger.debug(
            f"Rate limit check passed for user {identifier}. "
            f"Count: {new_count}/{self.max_requests} in current window."
        )

# --- Dependency Instance ---
# Create an instance of the RateLimiter class to use as a dependency
rate_limiter_dependency = RateLimiter()

# --- Example Usage (Informational) ---
# from fastapi import APIRouter, Depends
# from .rate_limiter import rate_limiter_dependency
#
# router = APIRouter() # Apply dependency per route or globally in main.py
#
# @router.get("/limited-resource", dependencies=[Depends(rate_limiter_dependency)])
# async def get_limited_resource():
#     # If code reaches here, rate limit check passed
#     return {"message": "You have access!"}