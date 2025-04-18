"""
Configuration Management for EVA Backend.

Loads settings from environment variables and .env files using pydantic-settings.
Provides a centralized way to access configuration values throughout the application.
"""

import logging
import os
from functools import lru_cache
from typing import List, Optional, Union # Added Union

from pydantic import Field, field_validator # Added field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Logger Setup ---
# Basic logging setup if not configured elsewhere (e.g., in main.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Environment Variable Handling ---
# Determine the environment (development, production, testing)
# Default to 'development' if not set
APP_ENV = os.getenv("APP_ENV", "development").lower()
logger.info(f"Application environment detected: {APP_ENV}")

# Load appropriate .env file based on environment
env_file = f".env.{APP_ENV}" if APP_ENV != "production" else ".env"
if not os.path.exists(env_file) and APP_ENV != "production":
    logger.warning(f"Environment file '{env_file}' not found. Trying default '.env'.")
    env_file = ".env" # Fallback to default .env if specific one not found

if os.path.exists(env_file):
    logger.info(f"Loading settings from environment file: {env_file}")
else:
    logger.warning(f"Environment file '{env_file}' not found. Relying solely on environment variables.")
    env_file = None # Set to None if no env file exists


# --- Settings Model ---
class Settings(BaseSettings):
    """
    Defines application settings, loaded from environment variables and .env files.
    """
    # --- Core Application Settings ---
    APP_NAME: str = "EVA Backend"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = Field(default=APP_ENV) # Use the detected environment
    DEBUG: bool = Field(default=False, description="Enable debug mode (more verbose logging, etc.)")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    # --- API / Server Settings ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = Field(default="EVA", description="Name of the project")
    # PORT environment variable is automatically used by Cloud Run, but good to define default
    PORT: int = Field(default=8080, description="Port the application listens on")
    HOST: str = Field(default="0.0.0.0", description="Host the application binds to")

    # --- Security Settings ---
    SECRET_KEY: str = Field(..., description="Secret key for JWT token generation (MUST be set)")
    ALGORITHM: str = Field(default="HS256", description="Algorithm for JWT token encoding")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24 * 7, description="Access token expiry time in minutes (default: 7 days)") # e.g., 7 days

    # --- CORS Settings ---
    # Default allows all origins for development, should be restricted in production
    CORS_ORIGINS: List[str] = Field(default=["*"], description="List of allowed origins for CORS")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])

    # --- Database Settings ---
    # Firestore specific (if using Firestore)
    FIREBASE_PROJECT_ID: Optional[str] = Field(default=None, description="Google Cloud Project ID for Firebase")
    # Path to service account key file (mounted as secret in Cloud Run)
    FIREBASE_CREDENTIALS_PATH: Optional[str] = Field(default="/app/secrets/firebase-credentials.json", description="Path to Firebase service account JSON file")
    # Or use default credentials if running in GCP environment with appropriate service account
    USE_GCP_DEFAULT_CREDENTIALS: bool = Field(default=False, description="Use default GCP credentials instead of a key file")

    # --- LLM Service Settings ---
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider to use (e.g., 'gemini', 'openai')")
    # Gemini Specific
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="API Key for Google Gemini")
    GEMINI_MODEL: str = Field(default="gemini-1.5-flash-latest", description="Specific Gemini model to use") # Or gemini-pro, etc.
    # OpenAI Specific (Example)
    # OPENAI_API_KEY: Optional[str] = Field(default=None, description="API Key for OpenAI")
    # OPENAI_MODEL: str = Field(default="gpt-4o", description="Specific OpenAI model to use")

    # --- LLM Generation Parameters ---
    LLM_TEMPERATURE: float = Field(default=0.7, description="LLM generation temperature (creativity)")
    LLM_MAX_TOKENS: int = Field(default=2048, description="Maximum tokens for LLM response") # Adjust based on model limits

    # --- Rate Limiting ---
    RATE_LIMIT_USER_REQUESTS: int = Field(default=100, description="Max requests per user per time window")
    RATE_LIMIT_USER_WINDOW_SECONDS: int = Field(default=60, description="Time window for user rate limiting (in seconds)")
    RATE_LIMIT_GLOBAL_REQUESTS: int = Field(default=1000, description="Max requests globally per time window")
    RATE_LIMIT_GLOBAL_WINDOW_SECONDS: int = Field(default=60, description="Time window for global rate limiting (in seconds)")
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable or disable rate limiting")

    # --- WebSocket Settings ---
    WEBSOCKET_MAX_QUEUE_SIZE: int = Field(default=10, description="Maximum number of messages to queue for a WebSocket client")
    WEBSOCKET_TIMEOUT_SECONDS: int = Field(default=60 * 10, description="Timeout for WebSocket connections (in seconds)") # 10 minutes

    # --- Memory Management ---
    MEMORY_DEFAULT_RETENTION_DAYS: Optional[int] = Field(default=None, description="Default retention period for memories (in days, None for indefinite)")
    MEMORY_SUMMARY_THRESHOLD: int = Field(default=10, description="Number of conversation turns before attempting summarization")
    MEMORY_SEARCH_LIMIT: int = Field(default=10, description="Default number of memories to retrieve in searches")

    # --- Caching ---
    CACHE_ENABLED: bool = Field(default=True, description="Enable or disable caching")
    CACHE_DEFAULT_TTL_SECONDS: int = Field(default=300, description="Default cache Time-To-Live (in seconds)") # 5 minutes

    # --- Validator for CORS_ORIGINS ---
    @field_validator("CORS_ORIGINS", mode='before')
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parses comma-separated string from env var into a list."""
        if isinstance(v, str):
            # Split comma-separated string and strip whitespace
            # Filter out empty strings that might result from trailing commas
            origins = [origin.strip() for origin in v.split(",") if origin.strip()]
            logger.debug(f"Parsing CORS_ORIGINS string '{v}' into list: {origins}")
            return origins
        elif isinstance(v, list):
            # It's already a list (like the default value), return as is
            logger.debug(f"CORS_ORIGINS is already a list: {v}")
            return v
        # Handle unexpected types if necessary, or raise error
        logger.error(f"Invalid type for CORS_ORIGINS: {type(v)}. Expected str or List[str].")
        raise ValueError("Invalid type for CORS_ORIGINS. Expected str or List[str].")


    # --- Pydantic Settings Configuration ---
    model_config = SettingsConfigDict(
        env_file=env_file,          # Specify the .env file to load
        env_file_encoding='utf-8',  # Encoding for the .env file
        extra='ignore',             # Ignore extra environment variables not defined in the model
        case_sensitive=False        # Environment variables are typically case-insensitive
    )

# --- Singleton Instance ---
# Use lru_cache to create a singleton instance of the Settings object
# This ensures settings are loaded only once.
@lru_cache()
def get_settings() -> Settings:
    """Returns the cached Settings instance."""
    logger.info("Initializing application settings...")
    try:
        _settings = Settings()
        # Optionally log some settings on startup (avoid logging secrets!)
        logger.info(f"APP_ENV: {_settings.APP_ENV}")
        logger.info(f"LLM_PROVIDER: {_settings.LLM_PROVIDER}")
        logger.info(f"GEMINI_MODEL: {_settings.GEMINI_MODEL}")
        # Be careful logging list values that might be sensitive if misinterpreted
        # logger.info(f"CORS_ORIGINS: {_settings.CORS_ORIGINS}")
        return _settings
    except Exception as e:
        logger.exception("CRITICAL: Failed to load application settings.", exc_info=e)
        # In a real application, you might want to exit if settings fail to load
        raise RuntimeError(f"Failed to load application settings: {e}")


# --- Example Usage (for testing if run directly) ---
if __name__ == "__main__":
    settings = get_settings()
    print("Settings loaded successfully:")
    print(f"  APP_NAME: {settings.APP_NAME}")
    print(f"  APP_ENV: {settings.APP_ENV}")
    print(f"  DEBUG: {settings.DEBUG}")
    print(f"  SECRET_KEY: {'*' * len(settings.SECRET_KEY) if settings.SECRET_KEY else 'Not Set'}") # Mask secret key
    print(f"  CORS_ORIGINS: {settings.CORS_ORIGINS}")
    print(f"  FIREBASE_PROJECT_ID: {settings.FIREBASE_PROJECT_ID}")
    print(f"  GEMINI_API_KEY: {'Set' if settings.GEMINI_API_KEY else 'Not Set'}")
    print(f"  GEMINI_MODEL: {settings.GEMINI_MODEL}")