"""
Utility functions for the EVA backend application.
"""

import logging
from datetime import datetime, timezone # Ensure timezone is imported
from typing import Optional

# pip install python-dateutil
try:
    from dateutil import parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

async def parse_datetime_from_text(text: str) -> Optional[datetime]:
    """
    Parses a datetime from text using dateutil.parser if available.
    Tries to find dates/times within the text.
    Returns timezone-aware datetime (UTC) if possible, otherwise naive.
    """
    if not DATEUTIL_AVAILABLE:
        logger.error("dateutil library not installed. Cannot parse datetime from text.")
        # Fallback logic (very basic)
        if "tomorrow" in text.lower():
             from datetime import timedelta # Local import
             # Return timezone-aware UTC time for tomorrow 9 AM
             return (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        return None

    try:
        # fuzzy=True helps find dates within surrounding text
        # default=datetime.now(timezone.utc) could provide context, but might be confusing
        # Try parsing, assuming it might contain timezone info
        dt_naive_or_aware = parser.parse(text, fuzzy=True)

        # Make it timezone-aware (assume UTC if naive)
        if dt_naive_or_aware.tzinfo is None or dt_naive_or_aware.tzinfo.utcoffset(dt_naive_or_aware) is None:
             # If naive, assume UTC
             dt_aware = dt_naive_or_aware.replace(tzinfo=timezone.utc)
             logger.debug(f"Parsed naive datetime '{dt_naive_or_aware}', assuming UTC: '{dt_aware}' from text: '{text}'")
        else:
             # If aware, convert to UTC
             dt_aware = dt_naive_or_aware.astimezone(timezone.utc)
             logger.debug(f"Parsed timezone-aware datetime '{dt_naive_or_aware}', converted to UTC: '{dt_aware}' from text: '{text}'")

        return dt_aware

    except (ValueError, OverflowError, TypeError) as e:
        # ValueError covers parsing failures, OverflowError for dates too far in past/future
        logger.debug(f"Could not parse datetime from text using dateutil: '{text}'. Error: {e}")
        # Optionally try LLM here as a fallback for complex cases?
        return None
    except Exception as e:
        # Catch any other unexpected errors during parsing
        logger.error(f"Unexpected error parsing datetime from text: '{text}'. Error: {e}", exc_info=True)
        return None

# Add other utility functions here as needed