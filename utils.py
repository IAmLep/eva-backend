"""
Utility functions for the EVA backend.

Includes functions for date/time parsing, potentially data validation,
and placeholder encryption/decryption functions.
"""

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Union

# Placeholder for dateutil if needed, or use standard library
try:
    from dateutil import parser as dateutil_parser
    from dateutil.tz import UTC
except ImportError:
    dateutil_parser = None
    UTC = timezone.utc # Fallback to standard library UTC

logger = logging.getLogger(__name__)

# --- Date/Time Parsing ---

async def parse_datetime_from_text(text: str) -> Optional[datetime]:
    """
    Attempts to parse a datetime object from various natural language formats in text.
    Uses dateutil.parser if available for robustness, otherwise basic regex.

    Args:
        text: The input string potentially containing date/time information.

    Returns:
        A timezone-aware datetime object (UTC) if parsing is successful, otherwise None.
    """
    now = datetime.now(UTC)

    # Simple regex patterns first (case-insensitive)
    # Examples: "tomorrow", "tonight", "next week", "in 2 hours", "at 5pm"
    text_lower = text.lower()

    if "tomorrow" in text_lower:
        match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text_lower)
        target_date = now.date() + timedelta(days=1)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            ampm = match.group(3)
            if ampm == 'pm' and hour != 12: hour += 12
            if ampm == 'am' and hour == 12: hour = 0 # Midnight case
            try:
                return datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=UTC)
            except ValueError: # Handle invalid hour/minute like 25:00
                 pass # Fall through to general parser
        else:
            # Default to 9 AM tomorrow if no time specified
            return datetime(target_date.year, target_date.month, target_date.day, 9, 0, tzinfo=UTC)

    if "tonight" in text_lower:
        match = re.search(r"(\d{1,2})(?::(\d{2}))?", text_lower)
        target_date = now.date() # Assume tonight means today's date
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2) or 0)
            # Assume PM if time is between 1 and 11
            if 1 <= hour < 12: hour += 12
            try:
                dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=UTC)
                # If resulting time is in the past, assume it's for tomorrow night
                if dt < now:
                    target_date += timedelta(days=1)
                    dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=UTC)
                return dt
            except ValueError:
                pass # Fall through
        else:
             # Default to 8 PM tonight
             dt = datetime(target_date.year, target_date.month, target_date.day, 20, 0, tzinfo=UTC)
             if dt < now: # If 8pm already passed today
                 target_date += timedelta(days=1)
                 dt = datetime(target_date.year, target_date.month, target_date.day, 20, 0, tzinfo=UTC)
             return dt

    # Relative times like "in 2 hours", "in 30 minutes"
    match = re.search(r"in (\d+)\s+(hour|minute)s?", text_lower)
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == "hour":
            return now + timedelta(hours=value)
        elif unit == "minute":
            return now + timedelta(minutes=value)

    # --- Use dateutil.parser if available (more robust) ---
    if dateutil_parser:
        try:
            # fuzzy=True helps find dates within longer strings
            # Ignoretz=False, tzinfos might be needed for local timezones if not UTC assumed
            parsed_dt = dateutil_parser.parse(text, fuzzy=True, ignoretz=False)

            # Make timezone-aware (assume UTC if naive)
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=UTC)
            else:
                # Convert to UTC if it has a different timezone
                parsed_dt = parsed_dt.astimezone(UTC)

            # Basic sanity check: Avoid dates too far in the past unless explicitly stated
            # (e.g., "last tuesday" - dateutil handles this, but check threshold)
            if parsed_dt < now - timedelta(days=365*5): # Arbitrary 5 years ago limit
                 logger.warning(f"Parsed date {parsed_dt} is very old, might be incorrect.")
                 # Decide whether to return it or None based on requirements

            # If the parsed time is in the past today, but the text implies future (e.g., "at 2pm"), advance day?
            # This logic can get complex. Relying on dateutil's intelligence for now.

            logger.debug(f"Dateutil parsed '{text}' as: {parsed_dt}")
            return parsed_dt
        except (ValueError, OverflowError, TypeError) as e:
            logger.debug(f"Dateutil failed to parse '{text}': {e}")
            # Fall through if dateutil fails
        except Exception as e:
            logger.error(f"Unexpected error using dateutil on '{text}': {e}", exc_info=True)


    # --- Fallback Regex for specific formats (e.g., "5pm", "14:30") ---
    # This is less robust than dateutil
    match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        ampm = match.group(3)
        target_date = now.date()

        if ampm and ampm.lower() == 'pm' and hour != 12: hour += 12
        if ampm and ampm.lower() == 'am' and hour == 12: hour = 0 # Midnight

        try:
            dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=UTC)
            # If the time has already passed today, assume it's for tomorrow
            if dt < now:
                target_date += timedelta(days=1)
                dt = datetime(target_date.year, target_date.month, target_date.day, hour, minute, tzinfo=UTC)
            return dt
        except ValueError:
            pass # Invalid hour/minute like 25:00

    logger.warning(f"Could not parse datetime from text: '{text}'")
    return None


# --- Placeholder Encryption/Decryption ---
# WARNING: These are NOT secure. Replace with a real cryptographic library like 'cryptography'.

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """
    Placeholder encryption function. Does NOT provide real security.
    Reverses the data as a trivial example.
    """
    logger.warning("Using placeholder encryption. THIS IS NOT SECURE.")
    # Example: Simple reverse (DO NOT USE IN PRODUCTION)
    return data[::-1] + b"_placeholder" # Add suffix to distinguish

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """
    Placeholder decryption function. Only works with the placeholder encryption.
    """
    logger.warning("Using placeholder decryption. THIS IS NOT SECURE.")
    # Example: Reverse the reverse (DO NOT USE IN PRODUCTION)
    if not encrypted_data.endswith(b"_placeholder"):
        raise ValueError("Data was not encrypted with the placeholder function.")
    original_data = encrypted_data[:-len(b"_placeholder")]
    return original_data[::-1]

# --- Other Utilities ---

# Add any other utility functions needed by your application here.
# For example, data validation helpers, string formatters, etc.
