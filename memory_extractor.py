# memory_extractor.py
import logging

logger = logging.getLogger(__name__)


def extract_key_info_entries(memory_summary: str):
    """
    Extracts key-value pairs from a memory summary string.
    Expects a format like:
       "Name: Chris; Greeting: Hello"
    Returns a list of tuples: [(label, value), ...]
    Returns an empty list if the input is None or empty.
    """
    if not memory_summary:
        logger.debug("Memory summary is empty. Returning empty list.")
        return []

    try:
        entries = memory_summary.split(";")
        result = []
        for entry in entries:
            parts = entry.split(":")
            if len(parts) >= 2:
                label = parts[0].strip()
                value = ":".join(parts[1:]).strip()
                if label and value:
                    result.append((label, value))
        return result
    except Exception as e:
        logger.error(f"Error extracting key info: {e}", exc_info=True)
        return []  # Return an empty list to avoid crashing the application


# Example Usage (for testing)
if __name__ == "__main__":
    summary = "Name: Chris; Greeting: Hello; Age: 30"
    extracted_info = extract_key_info_entries(summary)
    print(f"Extracted Info: {extracted_info}")

    summary_empty = None
    extracted_info_empty = extract_key_info_entries(summary_empty)
    print(f"Extracted Info (Empty): {extracted_info_empty}")

    summary_malformed = "Name: Chris; Greeting"
    extracted_info_malformed = extract_key_info_entries(summary_malformed)
    print(f"Extracted Info (Malformed): {extracted_info_malformed}")